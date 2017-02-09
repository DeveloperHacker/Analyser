from collections import namedtuple
from multiprocessing import Pool

import _pickle as pickle
import numpy as np
import tensorflow as tf

from utils import generator
from utils.method import JavaDoc
from variables import STATE_SIZE, RESOURCES, BATCHES


class Tags:
    next = "@next"


def join(javaDoc: JavaDoc) -> list:
    joined = [
        ("head", javaDoc.head),
        ("params", (" %s " % Tags.next).join(javaDoc.params)),
        ("variables", (" %s " % Tags.next).join(javaDoc.variables)),
        ("results", (" %s " % Tags.next).join(javaDoc.results)),
        # ("sees", (" %s " % Tags.next).join(javaDoc.sees)),
        # ("throws", (" %s " % Tags.next).join(javaDoc.throws))
    ]
    return joined


def vectorization(joined: list) -> np.ndarray:
    vector = []
    for label, text in joined:
        splited = text.split(" ")
        vector.append(len(splited) if len(splited) > 1 else int(splited[0] != ""))
    return np.asarray(vector)


def firstNMax(clusters: list, n: int):
    maxes = []
    indexes = []
    for i, data in clusters:
        length = len(data)
        if len(maxes) < n:
            maxes.append(length)
            indexes.append(i)
        else:
            i_minimum = np.argmin(maxes)
            minimum = maxes[i_minimum]
            if minimum < length:
                maxes[i_minimum] = length
                indexes[i_minimum] = i
    return indexes, maxes


def maximize(f: callable, a: float, b: float, epsilon=1e-6) -> tuple:
    fi = 2 / (1 + np.math.sqrt(5))
    counter = 0
    step = (b - a) * fi
    x1 = b - step
    x2 = a + step
    y1 = f(x1)
    y2 = f(x2)
    while abs(b - a) < epsilon and counter < 20:
        if y1 <= y2:
            a = x1
            x1 = x2
            x2 = a + (b - a) * fi
            y1 = y2
            y2 = f(x2)
        else:
            b = x2
            x2 = x1
            x1 = b - (b - a) * fi
            y2 = y1
            y1 = f(x1)
        counter += 1
    x = (a + b) / 2
    return x, f(x)


Triplet = namedtuple("Triplet", ["left", "right", "distance"])


def separate(tar_i, tar_value, data) -> list:
    return [Triplet(tar_i, i, np.linalg.norm(tar_value - value, 2)) for i, value in enumerate(data) if tar_i < i]


def chunks(l, n):
    for i in range(0, len(l), n):
        yield l[i:i + n]


def batching(methods: list, cluster_size: int):
    num_data = len(methods)
    num_clusters = num_data // cluster_size
    docs = [join(method.javaDoc) for i, method in enumerate(methods)]
    vectors = [(i, vectorization(joined)) for i, joined in enumerate(docs)]
    clusters = generator.KMeans(vectors, num_clusters)
    idx_batches = [chunk for cluster in clusters for chunk in chunks(cluster, cluster_size) if
                   len(chunk) == cluster_size]
    print("%d/%d" % (len(idx_batches) * cluster_size, num_data))
    batches = [[docs[i] for i in idx_batch] for idx_batch in idx_batches]
    return batches


# def cluster(methods: list, cluster_size: int):
#     num_data = len(methods)
#     num_clusters = int(num_data / cluster_size)
#     docs = [join(method.javaDoc) for i, method in enumerate(methods)]
#     vectors = {i: vectorization(joined) for i, joined in enumerate(docs)}
#     print(vectors)
#     clusterer = generator.XNeighbors(vectors)
#     relevant = []
#     relevant_indexes = []
#
#     def f(threshold: float):
#         global relevant, relevant_indexes
#         print(threshold)
#         relevant = clusterer(threshold)
#         relevant_indexes, first_max = firstNMax(relevant, num_clusters)
#         score = 0
#         fine = 0
#         for length in first_max:
#             minimum = min(cluster_size, length)
#             score += minimum
#             fine += 0 if minimum == length else length - cluster_size
#         return score - fine
#
#     print("maximize")
#     max_threshold, max_value = maximize(f, 0.0, 1.0, 1e-2)
#     print(max_threshold, max_value)
#     print(len(relevant))
#     print(len(relevant_indexes))


def constructRNNNet(methods: list):
    lstm = tf.nn.rnn_cell.LSTMCell(STATE_SIZE)
    rnn = lambda batch, state_fw, state_bw: tf.nn.bidirectional_rnn(
        lstm, lstm,
        batch,
        dtype=tf.float32,
        initial_state_fw=state_fw,
        initial_state_bw=state_bw
    )
