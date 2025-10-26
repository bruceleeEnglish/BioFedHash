import numpy as np
import requests, argparse, time, logging

import config
import utils

def divisive_norm_rows(X, eps=1e-8):
    n = np.linalg.norm(X, ord=2, axis=1, keepdims=True) + eps
    return X / n

def split_columns(X, total_clients, client_id):
    N = X.shape[1]
    split = np.linspace(0, N, total_clients + 1, dtype=int)
    s, e = split[client_id], split[client_id+1]
    return X[:, s:e]

def get_client_partitions(dataset, total_clients, client_id, split_seed=None):
    if dataset == 'mnist':
        part = utils.load_mnist_partition()
        # 归一化
        train_X = divisive_norm_rows(part['train_X'])
        db_X    = divisive_norm_rows(part['db_X'])
        q_X     = divisive_norm_rows(part['query_X'])
        db_y, q_y = part['db_y'], part['query_y']
        meta = {'eval_kind':'labels', 'db_y': db_y, 'query_y': q_y, 'topR': None}
    elif dataset == 'cifar10':
        part = utils.load_cifar10_partition()
        train_X = divisive_norm_rows(part['train_X'])
        db_X    = divisive_norm_rows(part['db_X'])
        q_X     = divisive_norm_rows(part['query_X'])
        db_y, q_y = part['db_y'], part['query_y']
        meta = {'eval_kind':'labels', 'db_y': db_y, 'query_y': q_y, 'topR': config.CIFAR_TOP_R}
    else:
        if not config.GLOVE_FILE:
            raise ValueError("Please set --glove_file for GloVe dataset.")
        part = utils.load_glove_partition(config.GLOVE_FILE, config.GLOVE_MAX_WORDS, config.GLOVE_TRAIN_SIZE, config.GLOVE_DB_SIZE, seed=split_seed)
        train_X = divisive_norm_rows(part['train_X'])
        db_X    = divisive_norm_rows(part['db_X'])
        q_X     = divisive_norm_rows(part['query_X'])
        meta = {'eval_kind':'gt', 'topR': config.GLOVE_TOP_R}
        if client_id == 0:
            gt = utils.build_glove_ground_truth(q_X, db_X, top_k=config.GLOVE_TOP_R, metric=config.GLOVE_GT_METRIC)
            meta['gt'] = gt
    train_local = split_columns(train_X.T, total_clients, client_id)
    db_local    = split_columns(db_X.T,    total_clients, client_id)
    q_local     = split_columns(q_X.T,     total_clients, client_id)
    return {
        'train_local': train_local,
        'db_local': db_local,
        'q_local': q_local,
        'hid_feat_dim': train_local.shape[0]
    }, meta

def client_train_local(W, data_subset):
    # data_subset: [feat_local, B]
    sign = np.sign(W)
    return np.dot(sign * np.absolute(W) ** (config.P - 1), data_subset)

def client_update(W, data_subset, tot_input, yl, eps=None):
    xx = np.sum(np.multiply(yl, tot_input), 1)
    ds = np.dot(yl, np.transpose(data_subset)) - np.multiply(np.tile(xx.reshape(xx.shape[0], 1), (1, data_subset.shape[0])), W)
    nc = np.amax(np.absolute(ds))
    if nc < config.PREC: nc = config.PREC
    step = (eps if eps is not None else config.EPS)
    return W + step * np.true_divide(ds, nc)

def generate_binary_projection_bernoulli(hid, feat_dim, p=0.1):
    return (np.random.rand(hid, feat_dim) < p).astype(np.float32)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--id', type=int, required=True)
    parser.add_argument('--dataset', type=str, choices=['mnist','cifar10','glove'], default=config.DATASET)
    parser.add_argument('--port', type=int, default=config.SERVER_PORT)
    parser.add_argument('--glove_file', type=str, default=None)
    parser.add_argument('--round_size', type=int, default=None)
    parser.add_argument('--impl', type=str, choices=['BioFedHash','BioFedHash++'], default=config.IMPL)
    args = parser.parse_args()

    config.DATASET = args.dataset
    config.SERVER_PORT = args.port
    if args.glove_file: config.GLOVE_FILE = args.glove_file
    if args.round_size: config.ROUND_SIZE = args.round_size
    config.IMPL = args.impl

    client_id = args.id
    total_clients = config.CLIENT_SIZE
    server_url = f"http://{config.SERVER_HOST}:{config.SERVER_PORT}"

    logging.basicConfig(level=logging.INFO, format=f'[%(asctime)s] [Client {client_id}] %(message)s', encoding='utf-8')

    session = requests.Session()
    adapter = requests.adapters.HTTPAdapter(pool_connections=8, pool_maxsize=8, max_retries=3)
    session.mount('http://', adapter)
    session.headers.update({'Connection': 'keep-alive'})

    try:
        session.post(f"{server_url}/register", json={'client_id': client_id}, timeout=(3, 20)).raise_for_status()
    except Exception as e:
        logging.error(f"register failed: {e}")
        return

    W = None
    data_pack = None
    feat_dim = None
    last_task = None
    meta_sent_for_seed = set()

    while True:
        try:
            r = session.get(f"{server_url}/get_task", timeout=(3, 20))
            if r.status_code == 410:
                logging.info("finished."); return
            if r.status_code == 202:
                time.sleep(0.3); continue
            task = r.json()
        except Exception as e:
            logging.error(f"get_task failed: {e}")
            time.sleep(0.5); continue

        phase = task.get('phase')
        k     = task.get('k')
        hid   = task.get('hid')
        seed  = task.get('seed', None)
        key   = (phase, k, hid, seed)

        if (data_pack is None) or (last_task is None) or (last_task[-1] != seed):
            data_pack, meta = get_client_partitions(config.DATASET, total_clients, client_id, split_seed=seed)
            feat_dim = data_pack['hid_feat_dim']
            W = None
            last_task = key

        if (W is None) or (W.shape != (hid, feat_dim)):
            if config.IMPL == 'BioFedHash':
                W = generate_binary_projection_bernoulli(hid, feat_dim, p=config.FEDHASH_BERNOULLI_P)
            else:
                W = np.random.normal(0.0, 1.0, (hid, feat_dim)).astype(np.float32)

        if phase == 'train':
            logging.info(f"Phase=train k={k} hid={hid} seed={seed}")
            if config.IMPL == 'BioFedHash':
                try:
                    session.post(f"{server_url}/notify_phase_done", json={'client_id': client_id, 'k': k, 'phase':'train'}, timeout=(3, 20)).raise_for_status()
                except Exception as e:
                    logging.error(f"notify train done failed: {e}")
                time.sleep(0.2)
                continue


            train_X_T = data_pack['train_local']
            num_batches_per_epoch = max(1, train_X_T.shape[1] // config.BATCH_SIZE)
            Tmax = 100

            for rnd in range(config.ROUND_SIZE):
                start = (config.BATCH_SIZE * rnd) % train_X_T.shape[1]
                idx = np.arange(start, start + config.BATCH_SIZE) % train_X_T.shape[1]
                subset = train_X_T[:, idx]

                try:
                    Z_local = client_train_local(W, subset)
                    payload = {'client_id': client_id, 'k': k, 'round': rnd, 'input': utils.numpy_to_base64(Z_local)}
                    session.post(f"{server_url}/send_input", json=payload, timeout=(3, 30)).raise_for_status()
                except Exception as e:
                    logging.error(f"send_input failed: {e}")
                    time.sleep(0.2); continue

                # 轮询竞争信号
                while True:
                    res = session.get(f"{server_url}/get_round_result", params={'k': k, 'round': rnd}, timeout=(3, 30))
                    if res.status_code == 200:
                        out = res.json()
                        tot_input = utils.base64_to_numpy(out['tot_input'])
                        yl = utils.base64_to_numpy(out['yl'])
                        break
                    elif res.status_code == 202:
                        time.sleep(0.1)
                    else:
                        time.sleep(0.15)

                epoch = rnd // num_batches_per_epoch
                eps_t = config.EPS * max(0.0, 1.0 - epoch / Tmax)
                W = client_update(W, subset, tot_input, yl, eps=eps_t)

                if (rnd+1) % 500 == 0:
                    logging.info(f"round {rnd+1}/{config.ROUND_SIZE}")


            try:
                session.post(f"{server_url}/notify_phase_done", json={'client_id': client_id, 'k': k, 'phase':'train'}, timeout=(3, 20)).raise_for_status()
            except Exception as e:
                logging.error(f"notify train done failed: {e}")

        elif phase in ('hash_db', 'hash_query'):
            is_db = (phase == 'hash_db')
            X_T = data_pack['db_local'] if is_db else data_pack['q_local']  # [feat_local, N]
            N = X_T.shape[1]
            B = config.BATCH_SIZE

            seed_tag = ('labels' if 'db_y' in meta or 'gt' not in meta else 'gt', seed)
            if seed_tag not in meta_sent_for_seed:
                try:
                    if meta.get('eval_kind') == 'labels':
                        payload = {
                            'eval_kind': 'labels',
                            'db_y': utils.numpy_to_base64(meta['db_y']),
                            'query_y': utils.numpy_to_base64(meta['query_y']),
                            'topR': meta.get('topR')
                        }
                    else:
                        if client_id == 0 and ('gt' in meta):
                            payload = {
                                'eval_kind': 'gt',
                                'gt': utils.numpy_to_base64(meta['gt']),
                                'topR': meta.get('topR', config.GLOVE_TOP_R)
                            }
                        else:
                            payload = None
                    if payload is not None:
                        session.post(f"{server_url}/send_meta", json=payload, timeout=(5, 60)).raise_for_status()
                        meta_sent_for_seed.add(seed_tag)
                except Exception as e:
                    logging.warning(f"send_meta failed (ignored if another client sent): {e}")


            num_batches = (N + B - 1) // B
            for bid in range(num_batches):
                s = bid * B
                e = min(N, s + B)
                subset = X_T[:, s:e]  # [feat_local, bsize]
                Z_local = client_train_local(W, subset)  # [hid, bsize]
                payload = {
                    'client_id': client_id,
                    'k': k,
                    'phase': phase,
                    'batch_id': bid,
                    'proj': utils.numpy_to_base64(Z_local)
                }
                try:
                    session.post(f"{server_url}/send_projection", json=payload, timeout=(5, 120)).raise_for_status()
                except Exception as e:
                    logging.error(f"send_projection failed: {e}")
                    time.sleep(0.2)


            try:
                session.post(f"{server_url}/notify_phase_done", json={'client_id': client_id, 'k': k, 'phase':phase}, timeout=(3, 20)).raise_for_status()
            except Exception as e:
                logging.error(f"notify phase done failed: {e}")

        else:
            time.sleep(0.2)
            continue

if __name__ == "__main__":
    main()