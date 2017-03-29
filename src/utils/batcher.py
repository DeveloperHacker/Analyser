import numpy as np

from utils import generator
from utils.wrapper import trace
from variables.tags import PARTS


def reshape(batch: list):
    new_batch = {label: [] for label in PARTS}
    for joined in batch:
        for label, embs in joined:
            new_batch[label].append(embs)
    return list(new_batch.items())


def vector(data: list) -> np.ndarray:
    return np.asarray([len(embeddings) for label, embeddings in data])


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


def chunks(line: list, block_size: int):
    for i in range(0, len(line), block_size):
        yield line[i:i + block_size]


def hist(data: list, basket_sizes: list, key=None) -> dict:
    basket_sizes = list(sorted(basket_sizes))
    baskets = {basket: [] for basket in basket_sizes}
    for datum in data:
        for i, right in enumerate(basket_sizes):
            left = basket_sizes[i - 1] if i > 0 else 0
            if left < datum if key is None else key(datum) < right:
                baskets[right].append(datum)
    return baskets


@trace
def throwing(data: list, basket_sizes: list) -> dict:
    key = lambda datum: max([len(embeddings) for label, embeddings in datum])
    return hist(data, basket_sizes, key=key)


@trace
def build_batches(data: list, cluster_size: int, key=lambda x: x):
    num_data = len(data)
    num_clusters = num_data // cluster_size
    vectors = [(i, vector(key(datum))) for i, datum in enumerate(data)]
    clusters = generator.KMeans(vectors, num_clusters)
    idx_batches = [chunk for cluster in clusters for chunk in chunks(cluster, cluster_size) if
                   len(chunk) == cluster_size]
    batches = [[data[i] for i in idx_batch] for idx_batch in idx_batches]
    batches = [reshape(batch) for batch in batches]
    return batches
