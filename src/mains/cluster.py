from utils import generator
from utils import printer
from utils import unpacker
from mains.variables import SAVE

numClusters = 2500

if __name__ == '__main__':
    _, embeddings = unpacker.unpackEmbeddings(SAVE, "1016865")
    clusters = generator.generateClusters(embeddings, numClusters)
    printer.printClusters(clusters, numClusters)
