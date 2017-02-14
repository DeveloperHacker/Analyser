import math

from mains.variables import FEATURES, EMB_STORAGE, EMB_MODEL
from utils import generator
from utils import printer
from utils import unpacker


def __norm(value):
    return value / math.sqrt(FEATURES)

if __name__ == '__main__':
    _, embeddings = unpacker.unpackEmbeddings(EMB_STORAGE, EMB_MODEL)
    clusters = generator.generateClustersXNeighbors(embeddings, 0.3)
    printer.XNeighbors(clusters)
