RESOURCES = "resources"

DATA_SETS = RESOURCES + "/data_sets"
SENTENCES = DATA_SETS + "/sentences.txt"
FILTERED = DATA_SETS + "/filtered.txt"
BATCHES = DATA_SETS + "/batches.pickle"
METHODS = DATA_SETS + "/methods.xml"
VEC_METHODS = DATA_SETS + "/methods.pickle"
EMBEDDINGS = DATA_SETS + "/word2vec.pickle"
CONTEXTS = DATA_SETS + "/contexts.pickle"

NETS = RESOURCES + "/nets"

WORD2VEC = NETS + "/word2vec"
WORD2VEC_MODEL = WORD2VEC + "/model.ckpt"
WORD2VEC_LOG = WORD2VEC + "/train.log"

SEQ2SEQ = NETS + "/seq2seq"
ANALYSER = SEQ2SEQ + "/analyser"
ANALYSER_MODEL = ANALYSER + "/model.ckpt"
Q_FUNCTION = SEQ2SEQ + "/q-function"
Q_FUNCTION_MODEL = Q_FUNCTION + "/model.ckpt"
SEQ2SEQ_LOG = SEQ2SEQ + "/train.log"
