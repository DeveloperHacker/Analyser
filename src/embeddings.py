import numpy as np
import tensorflow as tf
from tensorflow.models.embedding import word2vec_optimized as word2vec
from tensorflow.models.embedding.word2vec_optimized import Word2Vec
from utils import batcher
from utils import dumper
from utils import filter
from utils import generator
from utils import printer
from utils import unpacker
from utils.wrapper import trace
from variables import *

class W2VStorage:
    def __init__(self, options, w2i, i2w):
        self.options = options
        self.word2id = w2i
        self.id2word = i2w

@trace
def generate():
    methods = unpacker.unpackMethods()
    docs = filter.applyFiltersForMethods(methods)
    with open(FILTERED, "w") as file:
        file.write("\n".join((text for doc in docs for label, text in doc)))
    train()
    embeddings = restore()
    data = batcher.vectorization(docs, embeddings)
    baskets = batcher.throwing(data, [MAX_ENCODE_SEQUENCE])
    batches = {basket: batcher.batching(data, BATCH_SIZE) for basket, data in baskets.items()}
    dumper.dump(embeddings, EMBEDDINGS)
    dumper.dump(data, VEC_METHODS)
    dumper.dump(batches, BATCHES)


@trace
def train():
    word2vec.FLAGS.save_path = DATA_SETS
    word2vec.FLAGS.train_data = FILTERED
    word2vec.FLAGS.epochs_to_train = EMB_EPOCHS
    word2vec.FLAGS.embedding_size = EMB_SIZE
    word2vec.FLAGS.window_size = WINDOW
    options = word2vec.Options()

    with tf.Graph().as_default(), tf.Session() as session, tf.device("/cpu:0"):
        model = Word2Vec(options, session)
        for _ in range(options.epochs_to_train):
            model.train()
        model.saver.save(session, EMB_MODEL)
    dumper.dump(W2VStorage(model._options, model._word2id, model._id2word), EMB_STORAGE)


@trace
def restore() -> dict:
    storage = dumper.load(EMB_STORAGE)

    emb_dim = storage.options.emb_dim
    vocab_size = storage.options.vocab_size

    w_in = tf.Variable(tf.zeros([vocab_size, emb_dim]), name="w_in")
    w_out = tf.Variable(tf.zeros([vocab_size, emb_dim]), name="w_out")
    global_step = tf.Variable(0, name="global_step")

    saver = tf.train.Saver()

    with tf.Session() as sess:
        saver.restore(sess, EMB_MODEL)
        emb = w_in.eval(sess)  # type: np.multiarray.ndarray
        embeddings = {word.decode("utf8", errors='replace'): emb[i] for word, i in storage.word2id.items()}
    return embeddings


@trace
def cluster():
    embeddings = dumper.load(EMBEDDINGS)
    clusters = generator.KNeighbors(embeddings, 0.1)
    printer.XNeighbors(clusters)
