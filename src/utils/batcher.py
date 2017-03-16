import numpy as np

from utils import generator
from utils.wrapper import trace
from variables.tags import PARTS


def reshape(batch: list):
    new_batch = {label: ([], []) for label in PARTS}
    for joined in batch:
        for label, (embs, text) in joined:
            new_batch[label][0].append(embs)
            new_batch[label][1].append(text)
    return new_batch


def vector(data: list) -> np.ndarray:
    return np.asarray([len(embs) for label, (embs, text) in data])


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


@trace
def throwing(docs: list, basket_sizes: list):
    basket_sizes = list(sorted(basket_sizes))
    baskets = {basket: [] for basket in basket_sizes}
    for doc in docs:
        for i, right in enumerate(basket_sizes):
            left = basket_sizes[i - 1] if i > 0 else 0
            if left < max([len(embs) for label, (embs, text) in doc]) < right:
                baskets[right].append(doc)
    return baskets


@trace
def build_batches(docs: list, cluster_size: int):
    num_data = len(docs)
    num_clusters = num_data // cluster_size
    vectors = [(i, vector(doc)) for i, doc in enumerate(docs)]
    clusters = generator.KMeans(vectors, num_clusters)
    idx_batches = [chunk for cluster in clusters for chunk in chunks(cluster, cluster_size) if
                   len(chunk) == cluster_size]
    batches = [[docs[i] for i in idx_batch] for idx_batch in idx_batches]
    batches = [reshape(batch) for batch in batches]
    return batches
