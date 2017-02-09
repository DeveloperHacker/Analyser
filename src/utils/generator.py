import os
from collections import namedtuple
from multiprocessing.pool import Pool

from tensorflow.models.embedding import word2vec_optimized as word2vec
from tensorflow.models.embedding.word2vec_optimized import Word2Vec
from sklearn.cluster import KMeans
import _pickle as pickle
import tensorflow as tf
import numpy as np


class W2VStorage:
    def __init__(self, options, w2i, i2w):
        self.options = options
        self.word2id = w2i
        self.id2word = i2w


def generateTextSet(methods: list, absoluteFileName: str):
    sentences = []
    for method in methods:
        doc = []
        head = method.javaDoc.head  # type: str
        params = method.javaDoc.params  # type: list
        results = method.javaDoc.results  # type: list
        throws = method.javaDoc.throws  # type: list
        sees = method.javaDoc.sees  # type: list
        doc.append(head)
        doc.extend(params)
        doc.extend(results)
        doc.extend(throws)
        doc.extend(sees)
        sentences.extend([sentence for sentence in doc if len(sentence) > 0])
    with open(absoluteFileName, "w") as file:
        file.write("\n".join(sentences))
    return sentences


def generateEmbeddings(savePath: str, dataPath: str, epochs: int, features: int, window: int):
    word2vec.FLAGS.save_path = savePath
    word2vec.FLAGS.train_data = dataPath
    word2vec.FLAGS.epochs_to_train = epochs
    word2vec.FLAGS.embedding_size = features
    word2vec.FLAGS.window_size = window
    options = word2vec.Options()

    with tf.Graph().as_default(), tf.Session() as session, tf.device("/cpu:0"):
        model = Word2Vec(options, session)
        for _ in range(options.epochs_to_train):
            model.train()
        model.saver.save(session, os.path.join(options.save_path, "model.ckpt"), global_step=model.global_step)
    with open(os.path.join(options.save_path, 'JD2JDVs'), 'wb') as f:
        pickle.dump(W2VStorage(model._options, model._word2id, model._id2word), f)


def generateClustersKMeans(data: dict, numClusters: int):
    X = np.array(list(data.values()))
    kmeans = KMeans(n_clusters=numClusters, random_state=0).fit(X)
    clusters = [[] for _ in range(numClusters)]
    for word, vector in data.items():
        cluster = kmeans.predict([vector])
        clusters[cluster[0]].append(word)
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
