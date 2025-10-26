import os

# mnist | cifar10 | glove
DATASET = 'mnist'

# WTA | LTA
ALGORITHM = 'WTA'

# 'BioFedHash' | 'BioFedHash++'
IMPL = 'BioFedHash++'


FEDHASH_BERNOULLI_P = 0.1


P = 2.0
DELTA = 0.4
PREC = 1e-30
EPS = 2e-2


BATCH_SIZE = 100
ROUND_SIZE = 6000
CLIENT_SIZE = 2


K_HID_PAIRS = [
    (2, 400),
    (4, 800),
    (8, 1600),
    (16, 3200),
    (32, 6400),
]


CIFAR_TOP_R = 1000
GLOVE_TOP_R = 100
GLOVE_GT_METRIC = 'cosine'  # 'euclidean' | 'cosine'
EVAL_K_MULTIPLIERS = [1.0]


GLOVE_REPEATS = 10

SERVER_HOST = '127.0.0.1'
SERVER_PORT = 5000


GLOVE_FILE = os.environ.get('GLOVE_FILE', None)
GLOVE_MAX_WORDS = 50000
GLOVE_TRAIN_SIZE = 40000
GLOVE_DB_SIZE = 10000