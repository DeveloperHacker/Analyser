from utils import filter
from utils import generator
from utils import unpacker
from mains.variables import SAVE, DATA, METHODS, SENTENCES, FILTERED

EPOCHS = 500
FEATURES = 100
WINDOW = 5

if __name__ == '__main__':
    methods = unpacker.unpackMethods(METHODS)
    generator.generateTextSet(methods, SENTENCES)
    methods = filter.applyFiltersForMethods(methods)
    generator.generateTextSet(methods, FILTERED)
    generator.generateEmbeddings(SAVE, DATA, EPOCHS, FEATURES, WINDOW)
