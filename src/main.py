import os

import filters
import generators
import unpackers
import printers

RESOURCES_PATH = "../resources"
SAVE_PATH = RESOURCES_PATH + '/nets/javadoc->jvectors'
DATA_PATH = RESOURCES_PATH + '/sentencesFiltered.txt'
EPOCHS = 500
FEATURES = 100
WINDOW = 5

NUM_CLUSTERS = 2500

if __name__ == '__main__':
    """# Generate converter word from javaDoc to vectors
    astMethods = unpackers.unpackAstMethods(os.getcwd() + "/../resources/astMethods.xml")
    generators.generateTextSet(astMethods, os.getcwd() + "/../resources/sentences.txt")
    astMethods = filters.applyFiltersForMethods(astMethods)
    generators.generateTextSet(astMethods, os.getcwd() + "/../resources/filtered.txt")
    generators.generateEmbeddings(SAVE_PATH, DATA_PATH, EPOCHS, FEATURES, WINDOW)
    """

    """# Unpack embeddings for word from javaDoc
    _, embeddings = unpackers.unpackEmbeddings(SAVE_PATH, "1016865")
    """

    """# Clustering embedding
    clusters = generators.generateClusters(embeddings, NUM_CLUSTERS)
    printers.printClusters(clusters, NUM_CLUSTERS)
    """

    # Unpack daikonMethods and associate with astMethods
    astMethods = unpackers.unpackDaikonMethods(os.getcwd() + "/../resources/daikonMethods.xml")
    print("\n====================================================\n".join([str(astMethod) for astMethod in astMethods]))
