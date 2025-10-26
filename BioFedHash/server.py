import numpy as np
import threading, time, argparse, logging
from flask import Flask, request, jsonify

import config
import utils

app = Flask(__name__)
log = logging.getLogger('werkzeug'); log.setLevel(logging.ERROR)
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] [Server] %(message)s', encoding='utf-8')

state_lock = threading.Lock()
server_state = {
    'clients_registered': set(),
    'current_task': None,
    'round_data': {},
    'phase_done': {},
    'proj_buffers': {},
    'meta': {},
    'map_results': {},
    'training_finished': False
}

def server_aggregate_train(tot_client_input_list, hid, k, algorithm, delta):
    tot_input = sum(tot_client_input_list)                 # [hid, B]
    y = np.argsort(tot_input, axis=0)
    yl = np.zeros((hid, config.BATCH_SIZE), dtype=np.float32)
    if algorithm == 'WTA':
        yl[y[hid - 1, :], np.arange(config.BATCH_SIZE)] = 1.0
        if k > 1:
            yl[y[hid - k, :], np.arange(config.BATCH_SIZE)] = -delta
    else:
        yl[y[0, :], np.arange(config.BATCH_SIZE)] = 1.0
        if k > 1:
            yl[y[k - 1, :], np.arange(config.BATCH_SIZE)] = -delta
    return tot_input, yl

def wta_from_scores(scores, k_active, algorithm='WTA'):
    hid, B = scores.shape
    y = np.argsort(scores, axis=0)
    hc = np.full((hid, B), -1.0, dtype=np.float32)
    if algorithm == 'WTA':
        sel = y[hid - k_active:, :]
        hc[sel, np.arange(B)] = 1.0
    else:
        sel = y[:k_active, :]
        hc[sel, np.arange(B)] = 1.0
    return hc.T

def _get_phase_set(k, phase, create=True):
    key = (k, phase)
    if create:
        server_state['phase_done'].setdefault(key, set())
    return server_state['phase_done'].get(key, set())

def _get_proj_buf(k, phase):
    key = (k, phase)
    return server_state['proj_buffers'].setdefault(key, {'batches': {}, 'total_N': 0})

def training_orchestrator():
    logging.info("Waiting for clients to register ...")
    while True:
        with state_lock:
            if len(server_state['clients_registered']) >= config.CLIENT_SIZE: break
        time.sleep(0.5)
    logging.info(f"All {config.CLIENT_SIZE} clients registered. Start orchestrating.")

    repeats = config.GLOVE_REPEATS if config.DATASET == 'glove' else 1
    for rep in range(repeats):
        split_seed = np.random.randint(0, 2**31 - 1) if config.DATASET == 'glove' else None
        logging.info(f"=== Split {rep+1}/{repeats} seed={split_seed} ===")

        for k, hid in config.K_HID_PAIRS:
            with state_lock:
                server_state['current_task'] = {'phase':'train', 'k':k, 'hid':hid, 'seed': split_seed}
                server_state['round_data'].clear()
                server_state['phase_done'].pop((k, 'train'), None)
                server_state['proj_buffers'].pop((k, 'hash_db'), None)
                server_state['proj_buffers'].pop((k, 'hash_query'), None)
                server_state['meta'].clear()

            while True:
                with state_lock:
                    done = len(_get_phase_set(k, 'train', create=True)) == config.CLIENT_SIZE
                if done: break
                time.sleep(0.5)

            with state_lock:
                server_state['current_task'] = {'phase':'hash_db', 'k':k, 'hid':hid, 'seed': split_seed}
                server_state['phase_done'].pop((k, 'hash_db'), None)
            while True:
                with state_lock:
                    done = len(_get_phase_set(k, 'hash_db', create=True)) == config.CLIENT_SIZE
                if done: break
                time.sleep(0.5)

            with state_lock:
                server_state['current_task'] = {'phase':'hash_query', 'k':k, 'hid':hid, 'seed': split_seed}
                server_state['phase_done'].pop((k, 'hash_query'), None)
            while True:
                with state_lock:
                    done = len(_get_phase_set(k, 'hash_query', create=True)) == config.CLIENT_SIZE
                if done: break
                time.sleep(0.5)

            with state_lock:
                db_buf = _get_proj_buf(k, 'hash_db')
                q_buf  = _get_proj_buf(k, 'hash_query')
                meta   = dict(server_state['meta'])
                km_list = list(config.EVAL_K_MULTIPLIERS)

            db_batches = [db_buf['batches'][bid]['agg'] for bid in sorted(db_buf['batches'].keys())]
            q_batches  = [q_buf['batches'][bid]['agg'] for bid in sorted(q_buf['batches'].keys())]
            db_scores = np.concatenate(db_batches, axis=1) if db_batches else np.zeros((hid,0), dtype=np.float32)
            q_scores  = np.concatenate(q_batches,  axis=1) if q_batches else np.zeros((hid,0), dtype=np.float32)

            for km in km_list:
                k_eval = max(1, int(k * km))
                dbh = wta_from_scores(db_scores, k_eval, config.ALGORITHM)  # [N_db, hid]
                qh  = wta_from_scores(q_scores,  k_eval, config.ALGORITHM)  # [N_q,  hid]

                if meta.get('eval_kind') == 'labels':
                    db_y = meta.get('db_y'); q_y = meta.get('query_y'); topR = meta.get('topR', None)
                    score = utils.calculate_map_labels(qh, dbh, q_y, db_y, topR)
                elif meta.get('eval_kind') == 'gt':
                    gt = meta.get('gt'); topR = meta.get('topR', config.GLOVE_TOP_R)
                    score = utils.calculate_map_with_gt(qh, dbh, gt, topR)
                else:
                    logging.warning("No meta provided; skip mAP.")
                    score = 0.0

                with state_lock:
                    server_state['map_results'].setdefault(k, {}).setdefault(k_eval, []).append(score)
                logging.info(f"[split={rep+1}] [k={k}] mAP (k_eval={k_eval}): {score:.2f}%")

    with state_lock:
        server_state['training_finished'] = True
        server_state['current_task'] = None

    logging.info("=== All splits finished. Summary (mean over repeats) ===")
    with state_lock:
        for k, d in server_state['map_results'].items():
            for ke, arr in d.items():
                mean_score = float(np.mean(arr)) if len(arr) > 0 else 0.0
                logging.info(f"k_train={k}, k_eval={ke}: mean mAP={mean_score:.2f}% over {len(arr)} splits")

@app.route('/register', methods=['POST'])
def register():
    cid = request.json['client_id']
    with state_lock:
        server_state['clients_registered'].add(cid)
    return jsonify({'status': 'ok'}), 200

@app.route('/get_task', methods=['GET'])
def get_task():
    with state_lock:
        if server_state['training_finished']:
            return jsonify({'status': 'finished'}), 410
        task = server_state['current_task']
    if task is None:
        return jsonify({'status': 'waiting'}), 202
    return jsonify(task), 200

@app.route('/send_input', methods=['POST'])
def send_input():
    data = request.json
    cid = data['client_id']
    k = data['k']
    rnd = data['round']
    cin = utils.base64_to_numpy(data['input'])

    with state_lock:
        key = (k, rnd)
        if key not in server_state['round_data']:
            server_state['round_data'][key] = {'inputs': {}, 'result': None}
        server_state['round_data'][key]['inputs'][cid] = cin
        ready = len(server_state['round_data'][key]['inputs']) == config.CLIENT_SIZE
        if ready:
            cur = server_state['current_task']
            hid = cur['hid']
            inputs = [server_state['round_data'][key]['inputs'][i]
                      for i in sorted(server_state['round_data'][key]['inputs'].keys())]
        else:
            inputs = None

    if ready and inputs is not None:
        tot_input, yl = server_aggregate_train(inputs, hid, k, config.ALGORITHM, config.DELTA)
        with state_lock:
            server_state['round_data'][key]['result'] = {'tot_input': tot_input, 'yl': yl}

    return jsonify({'status': 'ok'}), 200

@app.route('/get_round_result', methods=['GET'])
def get_round_result():
    k = int(request.args.get('k'))
    rnd = int(request.args.get('round'))
    with state_lock:
        key = (k, rnd)
        rd = server_state['round_data'].get(key)
        res = None if rd is None else rd.get('result')
    if res is None:
        return jsonify({'status': 'processing'}), 202
    payload = {'tot_input': utils.numpy_to_base64(res['tot_input']), 'yl': utils.numpy_to_base64(res['yl'])}
    return jsonify(payload), 200

@app.route('/send_projection', methods=['POST'])
def send_projection():
    data = request.json
    cid   = data['client_id']
    k     = data['k']
    phase = data['phase']
    bid   = int(data['batch_id'])
    Zin   = utils.base64_to_numpy(data['proj'])

    with state_lock:
        buf = _get_proj_buf(k, phase)
        batches = buf['batches']
        if bid not in batches:
            batches[bid] = {'inputs': {}, 'agg': None}
        batches[bid]['inputs'][cid] = Zin
        ready = (len(batches[bid]['inputs']) == config.CLIENT_SIZE)

    if ready:
        with state_lock:
            inputs = [batches[bid]['inputs'][i] for i in sorted(batches[bid]['inputs'].keys())]
        agg = sum(inputs)
        with state_lock:
            batches[bid]['agg'] = agg
            if buf['total_N'] < 1:
                pass
    return jsonify({'status': 'ok'}), 200

@app.route('/notify_phase_done', methods=['POST'])
def notify_phase_done():
    data = request.json
    cid   = data['client_id']
    k     = data['k']
    phase = data['phase']
    with state_lock:
        _get_phase_set(k, phase, create=True).add(cid)
    return jsonify({'status':'ok'}), 200

@app.route('/send_meta', methods=['POST'])
def send_meta():
    data = request.json
    kind = data.get('eval_kind')
    meta = {}
    if kind == 'labels':
        meta['eval_kind'] = 'labels'
        meta['db_y'] = utils.base64_to_numpy(data['db_y'])
        meta['query_y'] = utils.base64_to_numpy(data['query_y'])
        meta['topR'] = data.get('topR')
    elif kind == 'gt':
        meta['eval_kind'] = 'gt'
        meta['gt'] = utils.base64_to_numpy(data['gt'])
        meta['topR'] = data.get('topR', config.GLOVE_TOP_R)
    with state_lock:
        server_state['meta'] = meta
    return jsonify({'status': 'ok'}), 200

@app.route('/send_weights', methods=['POST'])
def send_weights():
    return jsonify({'status': 'deprecated'}), 200

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, choices=['mnist','cifar10','glove'], default=config.DATASET)
    parser.add_argument('--algorithm', type=str, choices=['WTA','LTA'], default=config.ALGORITHM)
    parser.add_argument('--pairs', type=str, default=None, help='形如 "2:400,4:800"')
    parser.add_argument('--rounds', type=int, default=None)
    parser.add_argument('--eval_k_multipliers', type=str, default=None)
    parser.add_argument('--port', type=int, default=config.SERVER_PORT)
    parser.add_argument('--glove_repeats', type=int, default=None)
    args = parser.parse_args()

    config.DATASET = args.dataset
    config.ALGORITHM = args.algorithm
    if args.pairs:
        pairs=[]
        for it in args.pairs.split(','):
            k,h=it.split(':')
            pairs.append((int(k), int(h)))
        config.K_HID_PAIRS = pairs
    if args.rounds: config.ROUND_SIZE = args.rounds
    if args.eval_k_multipliers: config.EVAL_K_MULTIPLIERS = [float(x) for x in args.eval_k_multipliers.split(',')]
    if args.glove_repeats is not None: config.GLOVE_REPEATS = int(args.glove_repeats)
    config.SERVER_PORT = args.port

    logging.info(f"dataset={config.DATASET}, algo={config.ALGORITHM}, pairs={config.K_HID_PAIRS}, rounds={config.ROUND_SIZE}, repeats={config.GLOVE_REPEATS}")
    threading.Thread(target=training_orchestrator, daemon=True).start()
    app.run(host=config.SERVER_HOST, port=config.SERVER_PORT, threaded=True)