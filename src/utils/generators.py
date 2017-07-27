from collections import namedtuple
from multiprocessing.pool import Pool

import numpy as np
from sklearn.cluster import KMeans as _KMeans

from utils.wrappers import trace


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


class classifiers:
    @staticmethod
    @trace
    def kneighbors(data: dict, threshold: float):
        with Pool()as pool:
            state = [(label, target, data) for label, target in data.items()]
            divider = np.max(pool.starmap(maximum, state))
            state = pool.starmap(separate, [(*pair, divider, threshold) for pair in state])
        return state

    @staticmethod
    @trace
    def kmeans(data: list, numClusters: int):
        X = np.asarray([vector for _, vector in data])
        kmeans = _KMeans(n_clusters=numClusters, random_state=0, n_jobs=-1).fit(X)
        clusters = [[] for _ in range(numClusters)]
        for label, vector in data:
            clusters[kmeans.predict([vector])[0]].append(label)
        return clusters


class show:
    @staticmethod
    @trace
    def kmeans(clusters: list, num_clusters=None):
        if num_clusters is None:
            num_clusters = len(clusters)
        clusters = list(reversed(sorted(clusters, key=lambda cluster: len(cluster))))
        for i in range(0, round(num_clusters)):
            print(clusters[i])

    @staticmethod
    @trace
    def kneighbors(clusters: list, num_clusters=None):
        if num_clusters is None:
            num_clusters = len(clusters)
        clusters.sort(key=lambda target: len(target.data), reverse=True)
        for i in range(0, round(num_clusters)):
            print("\"%s\":" % clusters[i].label)
            for label, distance in clusters[i].data.items():
                print("\t%.3f: \"%s\"" % (distance, label))
