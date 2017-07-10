from multiprocessing.pool import Pool

import tensorflow as tf

from config import init
from constants.generator import WORD2VEC_EPOCHS, WINDOW, EMBEDDING_SIZE
from constants.paths import *
from utils import Dumper, Filter, Generator
from utils.wrapper import trace
from word2vec import word2vec_optimized as word2vec


def empty(method):
    for label, text in method["java-doc"].items():
        if len(text) > 0:
            return False
    return True


def join_java_doc(method):
    java_doc = {}
    for label, text in method["java-doc"].items():
        if label == "head":
            java_doc[label] = " ".join(text)
        else:
            java_doc[label] = (" %s " % Filter.NEXT).join(text)
    method["java-doc"] = java_doc
    return method


def extract_docs(method):
    result = (text.strip() for label, text in method["java-doc"].items())
    result = "\n".join(text for text in result if len(text) > 0)
    return result


def apply(method):
    if empty(method):
        return None
    method = Filter.apply(method)
    method = join_java_doc(method)
    doc = extract_docs(method)
    return doc


@trace
def prepare():
    methods = Dumper.json_load(ALL_METHODS)
    with Pool() as pool:
        docs = pool.map(apply, methods)
    docs = [doc for doc in docs if doc is not None]
    with open(FILTERED, "w") as file:
        file.write("\n".join(docs))


@trace
def generate():
    word2vec.FLAGS.save_path = GENERATOR
    word2vec.FLAGS.train_data = FILTERED
    word2vec.FLAGS.epochs_to_train = WORD2VEC_EPOCHS
    word2vec.FLAGS.embedding_size = EMBEDDING_SIZE
    word2vec.FLAGS.window_size = WINDOW
    options = word2vec.Options()

    with tf.Graph().as_default(), tf.Session() as session, tf.device("/cpu:0"):
        model = word2vec.Word2Vec(options, session)
        for _ in range(options.epochs_to_train):
            model.train()
        model.saver.save(session, GENERATOR_MODEL)
        emb = model.w_in.eval(session)
        embeddings = {word.decode("utf8", errors='replace'): emb[i] for word, i in model.word2id.items()}
    Dumper.pkl_dump(embeddings, EMBEDDINGS)


@trace
def cluster():
    embeddings = Dumper.pkl_load(EMBEDDINGS)
    clusters = Generator.classifiers.kneighbors(embeddings, 0.1)
    Generator.show.kneighbors(clusters)


if __name__ == '__main__':
    init()
    prepare()
    generate()
    cluster()
