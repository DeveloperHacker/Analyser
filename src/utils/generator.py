from collections import namedtuple
from multiprocessing.pool import Pool

from tensorflow.models.embedding import word2vec_optimized as word2vec
from tensorflow.models.embedding.word2vec_optimized import Word2Vec
from sklearn.cluster import KMeans as _KMeans
import _pickle as pickle
import tensorflow as tf
import numpy as np


class W2VStorage:
    def __init__(self, options, w2i, i2w):
        self.options = options
        self.word2id = w2i
        self.id2word = i2w


def generateEmbeddings(save_path: str, model_path: str, storage_path: str, data_path: str, epochs: int, features: int, window: int):
    word2vec.FLAGS.save_path = save_path
    word2vec.FLAGS.train_data = data_path
    word2vec.FLAGS.epochs_to_train = epochs
    word2vec.FLAGS.embedding_size = features
    word2vec.FLAGS.window_size = window
    options = word2vec.Options()

    with tf.Graph().as_default(), tf.Session() as session, tf.device("/cpu:0"):
        model = Word2Vec(options, session)
        for _ in range(options.epochs_to_train):
            model.train()
        model.saver.save(session, model_path)
    with open(storage_path, 'wb') as f:
        pickle.dump(W2VStorage(model._options, model._word2id, model._id2word), f)


def vectorization(joined: list, embeddings: dict):
    emb = lambda word: embeddings[word] if word in embeddings else embeddings['UNK']
    data = []
    for doc in joined:
        datum = []
        for label, text in doc:
            split = text.split(" ")
            datum.append((label, ([emb(word) for word in split], split)))
        data.append(datum)
    return data


def KMeans(data: list, numClusters: int):
    X = np.asarray([vector for _, vector in data])
    kmeans = _KMeans(n_clusters=numClusters, random_state=0, n_jobs=-1).fit(X)
    clusters = [[] for _ in range(numClusters)]
    for label, vector in data:
        clusters[kmeans.predict([vector])[0]].append(label)
    return clusters


def maximum(_, tar_value, data):
    result = [np.linalg.norm(tar_value - value, 2) for value in data.values()]
    return np.max(result)


StateParam = namedtuple('StateParam', ['label', 'data'])


def separate(tar_label, tar_value, data, divider, threshold) -> StateParam:
    result = {}
    for label, value in data.items():
        if label == tar_label: continue
        norm = np.linalg.norm(tar_value - value, 2) / divider
        if norm < threshold: result[label] = norm
    return StateParam(tar_label, result)


def XNeighbors(data: dict):
    def instance(threshold: float):
        # data = {k: v for k, v in list(data.items())[:10]}
        print("[XNeighbors]")
        pool = Pool()
        state = [(label, target, data) for label, target in data.items()]
        print("[MAXIMUM]")
        divider = np.max(pool.starmap(maximum, state))
        print("[SEPARATE]")
        state = pool.starmap(separate, [(*pair, divider, threshold) for pair in state])
        return state

    return instance


def generateClustersXNeighbors(data: dict, threshold: float):
    return XNeighbors(data)(threshold)
