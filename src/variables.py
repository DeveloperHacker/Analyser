RESOURCES = "resources"

DATA_SETS = RESOURCES + "/data_sets"
SENTENCES = DATA_SETS + "/sentences.txt"
FILTERED = DATA_SETS + "/filtered.txt"
BATCHES = DATA_SETS + "/batches.pickle"
METHODS = DATA_SETS + "/methods.xml"
VEC_METHODS = DATA_SETS + "/methods.pickle"
EMBEDDINGS = DATA_SETS + "/word2vec.pickle"

NETS = RESOURCES + "/nets"

WORD2VEC = NETS + "/word2vec"
WORD2VEC_MODEL = WORD2VEC + "/model.ckpt"
WORD2VEC_LOG = WORD2VEC + "/train.log"

SEQ2SEQ = NETS + "/seq2seq"
SEQ2SEQ_MODEL = SEQ2SEQ + "/model.ckpt"
SEQ2SEQ_LOG = SEQ2SEQ + "/train.log"

EMB_EPOCHS = 500
SEQ2SEQ_EPOCHS = 2000
EMB_SIZE = 100
WINDOW = 7
STATE_SIZE = 200
BATCH_SIZE = 20
MAX_ENCODE_SEQUENCE = 30
MAX_DECODE_SEQUENCE = 10
L2_WEIGHT = 0.0001
