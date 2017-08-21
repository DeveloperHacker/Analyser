import tensorflow as tf

import prepares
from contants import JAVA_DOC, EMBEDDINGS_PATH
from utils import dumpers, generators
from utils.wrappers import trace
from word2vec import word2vec_optimized as word2vec

flags = tf.app.flags

flags.DEFINE_bool('prepare', False, '')
flags.DEFINE_bool('train', False, '')
flags.DEFINE_bool('test', False, '')


@trace("PREPARE DATA-SET")
def prepare():
    methods = prepares.load(DATA_SET_PATH)
    methods = prepares.java_doc(methods)
    docs = (method[JAVA_DOC] for method in methods)
    with open(FLAGS.train_data, "w") as file:
        file.write("\n".join(docs))


@trace("TRAIN NET")
def train():
    options = word2vec.Options()
    config = tf.ConfigProto()
    config.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1
    with tf.Graph().as_default(), tf.Session(config=config) as session, tf.device("/cpu:0"):
        model = word2vec.Word2Vec(options, session)
        for _ in range(options.epochs_to_train):
            model.train()
        model.saver.save(session, MODEL_PATH)
        emb = model.w_in.eval(session)
        embeddings = {word.decode("utf8", errors='replace'): emb[i] for word, i in model.word2id.items()}
    dumpers.pkl_dump(embeddings, EMBEDDINGS_PATH)


@trace("TEST")
def test():
    embeddings = dumpers.pkl_load(EMBEDDINGS_PATH)
    clusters = generators.classifiers.kneighbors(embeddings, 0.1)
    generators.show.kneighbors(clusters)


FLAGS = flags.FLAGS

FLAGS.save_path = 'resources/word2vec'
FLAGS.train_data = 'resources/word2vec/filtered.txt'
FLAGS.epochs_to_train = 500
FLAGS.embedding_size = 100
FLAGS.window_size = 7

MODEL_PATH = 'resources/word2vec/model.ckpt'
DATA_SET_PATH = 'resources/data-sets/data-set.json'

if FLAGS.prepare: prepare()
if FLAGS.train: train()
if FLAGS.test: test()
