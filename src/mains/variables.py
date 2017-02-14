import os

RESOURCES = os.getcwd() + "/../../resources"

DATA_SETS = RESOURCES + "/data_sets"
SENTENCES = DATA_SETS + "/sentences.txt"
FILTERED = DATA_SETS + "/filtered.txt"
BATCHES = DATA_SETS + "/batches.pickle"
METHODS = DATA_SETS + "/methods.xml"
VEC_METHODS = DATA_SETS + "/methods.pickle"

NETS = RESOURCES + "/nets"

EMB = NETS + "/embeddings"
EMB_MODEL = EMB + "/model.ckpt"
EMB_STORAGE = EMB + "/model.storage"

SEQ2SEQ = NETS + "/seq2seq"

EPOCHS = 500
FEATURES = 100
WINDOW = 7
STATE_SIZE = 1000
BATCH_SIZE = 20
MAX_DECODE_SEQUENCE = 10
