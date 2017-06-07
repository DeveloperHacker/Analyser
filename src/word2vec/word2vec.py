import numpy as np
import tensorflow as tf

from constants.generator import WORD2VEC_EPOCHS, WINDOW, EMBEDDING_SIZE
from constants.paths import *
from utils import dumper, generator, printer
from utils.wrapper import trace
from word2vec import word2vec_optimized as word2vec


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
        model.saver.save(session, GENERATOR_MODEL, )
        emb = model.w_in.eval(session)  # type: np.multiarray.ndarray
        embeddings = {word.decode("utf8", errors='replace'): emb[i] for word, i in model.word2id.items()}
    dumper.dump(embeddings, EMBEDDINGS)


@trace
def cluster():
    embeddings = dumper.load(EMBEDDINGS)
    clusters = generator.KNeighbors(embeddings, 0.1)
    printer.XNeighbors(clusters)
