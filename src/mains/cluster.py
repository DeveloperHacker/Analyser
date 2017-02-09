import math

from utils import generator
from utils import printer
from utils import unpacker
from variables import SAVE, FEATURES


def __norm(value):
    return value / math.sqrt(FEATURES)

if __name__ == '__main__':
    ID = "21624603"
    _, embeddings = unpacker.unpackEmbeddings(SAVE, ID)
    clusters = generator.generateClustersXNeighbors(embeddings, 0.3)
    printer.XNeighbors(clusters)
