import numpy as np
import tensorflow as tf

from utils import batcher, dumper, filter, generator, printer, unpacker
from utils.wrapper import trace
from variables.path import *
from variables.train import *
from word2vec import word2vec_optimized as word2vec


@trace
def train():
    methods = unpacker.unpackMethods()
    docs = filter.applyFiltersForMethods(methods)
    with open(FILTERED, "w") as file:
        file.write("\n".join((text for doc in docs for label, text in doc if len(text) > 0)))
    embeddings = generate()
    dumper.dump(embeddings, EMBEDDINGS)
    data = batcher.vectorization(docs, embeddings)
    dumper.dump(data, VEC_METHODS)


@trace
def generate():
    word2vec.FLAGS.save_path = DATA_SETS
    word2vec.FLAGS.train_data = FILTERED
    word2vec.FLAGS.epochs_to_train = WORD2VEC_EPOCHS
    word2vec.FLAGS.embedding_size = EMBEDDING_SIZE
    word2vec.FLAGS.window_size = WINDOW
    options = word2vec.Options()

    with tf.Graph().as_default(), tf.Session() as session, tf.device("/cpu:0"):
        model = word2vec.Word2Vec(options, session)
        for _ in range(options.epochs_to_train):
            model.train()
        model.saver.save(session, WORD2VEC_MODEL, )
        emb = model.w_in.eval(session)  # type: np.multiarray.ndarray
        embeddings = {word.decode("utf8", errors='replace'): emb[i] for word, i in model.word2id.items()}
    return embeddings


@trace
def cluster():
    embeddings = dumper.load(EMBEDDINGS)
    clusters = generator.KNeighbors(embeddings, 0.1)
    printer.XNeighbors(clusters)


@trace
def run(foo: str):
    if foo == "train":
        train()
    elif foo == "cluster":
        cluster()