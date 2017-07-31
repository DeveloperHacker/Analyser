from multiprocessing.pool import Pool

import tensorflow as tf

from configurations.constants import TRAIN_EPOCHS, WINDOW_SIZE, EMBEDDING_SIZE
from configurations.paths import *
from configurations.tags import VARIABLE
from utils import dumpers, anonymizers, generators
from utils.wrappers import Timer
from word2vec import word2vec_optimized as word2vec


def empty(method):
    for label, text in method["java-doc"].items():
        if len(text) > 0:
            return False
    return True


def join_java_doc(method):
    params = (param["name"] for param in method["description"]["parameters"])
    java_doc = {VARIABLE: " ".join(["%s%d %s" % (VARIABLE, i, name) for i, name in enumerate(params)])}
    for label, text in method["java-doc"].items():
        java_doc[label] = " ".join(text)
    method["java-doc"] = java_doc
    return method


def extract_docs(method):
    result = (text.strip() for label, text in method["java-doc"].items())
    result = "\n".join(text for text in result if len(text) > 0)
    return result


def apply(method):
    if empty(method):
        return None
    method = join_java_doc(method)
    method = anonymizers.apply(method)
    doc = extract_docs(method)
    return doc


def prepare_data_set():
    methods = dumpers.json_load(FULL_DATA_SET)
    with Pool() as pool:
        docs = pool.map(apply, methods)
    docs = [doc for doc in docs if doc is not None]
    with open(FILTERED, "w") as file:
        file.write("\n".join(docs))


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


def cluster():
    embeddings = dumpers.pkl_load(EMBEDDINGS)
    clusters = generators.classifiers.kneighbors(embeddings, 0.1)
    generators.show.kneighbors(clusters)


if __name__ == '__main__':
    with Timer("PREPARE"):
        prepare_data_set()
    with Timer("TRAIN"):
        train()
    with Timer("TEST"):
        cluster()
