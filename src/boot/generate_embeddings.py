import tensorflow as tf

from boot import prepares
from configurations.constants import TRAIN_EPOCHS, WINDOW_SIZE, EMBEDDING_SIZE
from configurations.fields import JAVA_DOC
from configurations.paths import *
from utils import dumpers, generators
from utils.wrappers import trace
from word2vec import word2vec_optimized as word2vec


@trace("PREPARE DATA-SET")
def prepare_data_set():
    methods = dumpers.json_load(FULL_DATA_SET)
    methods = prepares.java_doc(methods)
    docs = (method[JAVA_DOC] for method in methods)
    with open(FILTERED, "w") as file:
        file.write("\n".join(docs))


@trace("TRAIN NET")
def train():
    word2vec.FLAGS.save_path = GENERATOR
    word2vec.FLAGS.train_data = FILTERED
    word2vec.FLAGS.epochs_to_train = TRAIN_EPOCHS
    word2vec.FLAGS.embedding_size = EMBEDDING_SIZE
    word2vec.FLAGS.window_size = WINDOW_SIZE
    options = word2vec.Options()

    config = tf.ConfigProto()
    config.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1
    with tf.Graph().as_default(), tf.Session(config=config) as session, tf.device("/cpu:0"):
        model = word2vec.Word2Vec(options, session)
        for _ in range(options.epochs_to_train):
            model.train()
        model.saver.save(session, GENERATOR_MODEL)
        emb = model.w_in.eval(session)
        embeddings = {word.decode("utf8", errors='replace'): emb[i] for word, i in model.word2id.items()}
    dumpers.pkl_dump(embeddings, EMBEDDINGS)


@trace("TEST")
def cluster():
    embeddings = dumpers.pkl_load(EMBEDDINGS)
    clusters = generators.classifiers.kneighbors(embeddings, 0.1)
    generators.show.kneighbors(clusters)


if __name__ == '__main__':
    prepare_data_set()
    train()
    cluster()
