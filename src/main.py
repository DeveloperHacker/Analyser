import os

import filter
import generator
import unpacker
import printer

RESOURCES_PATH = "../resources"
SAVE_PATH = RESOURCES_PATH + '/nets/javadoc->jvectors'
DATA_PATH = RESOURCES_PATH + '/dataSets/filtered.txt'
EPOCHS = 500
FEATURES = 100
WINDOW = 5

NUM_CLUSTERS = 2500

if __name__ == '__main__':
    # Unpack astMethods
    methods = unpacker.unpackMethods(os.getcwd() + "/" + RESOURCES_PATH + "/methods.xml")
    for method in methods:
        print(method)
        print()

    # # Generate converter word from javaDoc to vectors
    # generators.generateTextSet(methods, os.getcwd() + "/" + RESOURCES_PATH + "/sentences.txt")
    # methods = filters.applyFiltersForMethods(methods)
    # generators.generateTextSet(methods, os.getcwd() + "/" + RESOURCES_PATH + "/filtered.txt")
    # generators.generateEmbeddings(SAVE_PATH, DATA_PATH, EPOCHS, FEATURES, WINDOW)
    #
    # # Unpack embeddings for word from javaDoc
    # _, embeddings = unpackers.unpackEmbeddings(SAVE_PATH, "1016865")
    #
    # # Clustering embedding
    # clusters = generators.generateClusters(embeddings, NUM_CLUSTERS)
    # printers.printClusters(clusters, NUM_CLUSTERS)
