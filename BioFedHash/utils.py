import numpy as np
from scipy.spatial import distance
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm
import pickle, base64

def numpy_to_base64(arr):
    return base64.b64encode(pickle.dumps(arr)).decode('utf-8')

def base64_to_numpy(b64_string):
    return pickle.loads(base64.b64decode(b64_string.encode('utf-8')))

def hashing_from_scores(scores, k_active, algorithm='WTA'):
    # 输入 scores[h, N]，输出 hash[h, N] in {-1, +1}
    hid, Num = scores.shape
    y = np.argsort(scores, axis=0)
    hashcodes = np.full((hid, Num), -1.0, dtype=np.float32)
    if algorithm == 'WTA':
        sel = y[hid - k_active:, :]
        hashcodes[sel, np.arange(Num)] = 1.0
    else:
        sel = y[:k_active, :]
        hashcodes[sel, np.arange(Num)] = 1.0
    return hashcodes

def calculate_map_labels(query_hashes, db_hashes, query_labels, db_labels, top_R=None):
    AP_list = []
    for i in range(query_hashes.shape[0]):
        qh = query_hashes[i]
        dists = np.count_nonzero(qh != db_hashes, axis=1)
        ranked = np.argsort(dists)
        if top_R is not None:
            ranked = ranked[:top_R]
        ranked_labels = db_labels[ranked]
        y = query_labels[i]
        relevant = (ranked_labels == y)
        total_rel = np.sum(db_labels == y) if top_R is None else np.sum(relevant)
        if total_rel == 0:
            AP_list.append(0.0)
            continue
        num_rel, prec_sum = 0, 0.0
        for rank, is_rel in enumerate(relevant, 1):
            if is_rel:
                num_rel += 1
                prec_sum += num_rel / rank
        AP_list.append(prec_sum / max(1, total_rel))
    return float(np.mean(AP_list)) * 100.0

def calculate_map_with_gt(query_hashes, db_hashes, ground_truth_indices, top_R=100):
    AP_list = []
    for i in range(query_hashes.shape[0]):
        qh = query_hashes[i]
        dists = np.count_nonzero(qh != db_hashes, axis=1)
        ranked = np.argsort(dists)[:top_R]
        gt_set = set(ground_truth_indices[i])
        denom = len(gt_set)
        if denom == 0:
            AP_list.append(0.0)
            continue
        num_rel, prec_sum = 0, 0.0
        for rank, idx in enumerate(ranked, 1):
            if idx in gt_set:
                num_rel += 1
                prec_sum += num_rel / rank
        AP_list.append(prec_sum / denom)
    return float(np.mean(AP_list)) * 100.0

def draw_weights(synapses, Kx, Ky, outpath='weights_grid.png'):
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(10,10))
    yy=0
    HM=np.zeros((28*Ky,28*Kx))
    for y in range(Ky):
        for x in range(Kx):
            if yy < synapses.shape[0] and synapses.shape[1] >= 28*28:
                HM[y*28:(y+1)*28,x*28:(x+1)*28]=synapses[yy,:28*28].reshape(28,28)
                yy += 1
    plt.clf()
    if np.max(np.abs(HM)) > 0:
        nc=np.amax(np.absolute(HM))
        im=plt.imshow(HM,cmap='bwr',vmin=-nc/7,vmax=nc/7)
        fig.colorbar(im,ticks=[np.amin(HM), 0, np.amax(HM)])
    plt.axis('off')
    plt.savefig(outpath, dpi=300, bbox_inches='tight')

# 数据装载（客户端使用）
def load_mnist_partition():
    from torchvision.datasets import MNIST
    train = MNIST(root='./data', train=True, download=True)
    test = MNIST(root='./data', train=False, download=True)
    X = np.concatenate((train.data.numpy(), test.data.numpy()), axis=0)
    y = np.concatenate((train.targets.numpy(), test.targets.numpy()), axis=0)
    rng = np.random.RandomState()
    query_idx = []
    for c in range(10):
        idx = np.where(y == c)[0]
        rng.shuffle(idx)
        query_idx.extend(idx[:100])
    mask = np.zeros(len(X), dtype=bool)
    mask[query_idx] = True
    query_X = X[mask].reshape(-1, 28*28) / 255.0
    query_y = y[mask]
    db_X = X[~mask].reshape(-1, 28*28) / 255.0
    db_y = y[~mask]
    train_X = db_X
    return {'train_X': train_X, 'db_X': db_X, 'db_y': db_y, 'query_X': query_X, 'query_y': query_y}

def load_cifar10_partition():
    from tensorflow.keras.datasets import cifar10
    (train_X, train_y), (test_X, test_y) = cifar10.load_data()
    train_X = train_X.reshape(train_X.shape[0], -1).astype(np.float32)
    test_X = test_X.reshape(test_X.shape[0], -1).astype(np.float32)
    scaler = StandardScaler()
    train_X = scaler.fit_transform(train_X)
    test_X = scaler.transform(test_X)
    train_y = train_y.flatten()
    test_y = test_y.flatten()
    return {'train_X': train_X, 'db_X': train_X, 'db_y': train_y, 'query_X': test_X, 'query_y': test_y}

def load_glove_partition(glove_file, max_words=50000, train_size=40000, db_size=10000, seed=None):
    words, embs = [], []
    with open(glove_file, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= max_words: break
            vals = line.strip().split()
            if len(vals) < 2: continue
            words.append(vals[0])
            embs.append(np.asarray(vals[1:], dtype=np.float32))
    E = np.vstack(embs)
    if seed is None:
        rng = np.random
    else:
        rng = np.random.RandomState(seed)
    idx = np.arange(E.shape[0])
    rng.shuffle(idx)
    db_idx = idx[:db_size]
    train_idx = idx[db_size:db_size+train_size]
    db_raw = E[db_idx]
    train_raw = E[train_idx]
    scaler = StandardScaler(with_std=False)
    train_X = scaler.fit_transform(train_raw)
    db_X = scaler.transform(db_raw)
    query_X = db_X
    return {'train_X': train_X, 'db_X': db_X, 'query_X': query_X}

def build_glove_ground_truth(query_X, db_X, top_k=100, metric='cosine'):
    nn = NearestNeighbors(n_neighbors=top_k+1, metric=metric, n_jobs=-1)
    nn.fit(db_X)
    _, indices = nn.kneighbors(query_X)
    return indices[:, 1:]