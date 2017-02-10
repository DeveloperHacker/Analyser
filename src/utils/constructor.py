from collections import namedtuple
from multiprocessing import Pool

import _pickle as pickle
import numpy as np
import tensorflow as tf

from utils import generator
from utils.filter import Filter
from utils.method import JavaDoc
from variables import STATE_SIZE, RESOURCES, BATCHES


def join(javaDoc: JavaDoc) -> list:
    joined = [
        ("head", javaDoc.head),
        ("params", (" %s " % Filter.next).join(javaDoc.params)),
        ("variables", (" %s " % Filter.next).join(javaDoc.variables)),
        ("results", (" %s " % Filter.next).join(javaDoc.results)),
        # ("sees", (" %s " % Filter.next).join(javaDoc.sees)),
        # ("throws", (" %s " % Filter.next).join(javaDoc.throws))
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


def chunks(line, n):
    for i in range(0, len(line), n):
        yield line[i:i + n]


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


def constructRNNNet(batches: list):
    initializer = tf.truncated_normal_initializer(mean=0.0, stddev=0.01)
    lstm = tf.nn.rnn_cell.LSTMCell(STATE_SIZE)
    rnn = lambda batch, state_fw, state_bw: tf.nn.bidirectional_rnn(
        lstm, lstm,
        batch,
        dtype=tf.float32,
        initial_state_fw=state_fw,
        initial_state_bw=state_bw
    )

