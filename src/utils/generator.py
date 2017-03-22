from collections import namedtuple
from multiprocessing.pool import Pool

import numpy as np
from sklearn.cluster import KMeans as _KMeans

from utils.wrapper import trace


@trace
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


def KNeighborsClusterisator(data: dict):
    def instance(threshold: float):
        print("[XNeighbors]")
        pool = Pool()
        state = [(label, target, data) for label, target in data.items()]
        print("[MAXIMUM]")
        divider = np.max(pool.starmap(maximum, state))
        print("[SEPARATE]")
        state = pool.starmap(separate, [(*pair, divider, threshold) for pair in state])
        return state

    return instance


@trace
def KNeighbors(data: dict, threshold: float):
    return KNeighborsClusterisator(data)(threshold)
