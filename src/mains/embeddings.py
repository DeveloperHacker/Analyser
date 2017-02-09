from utils import filter
from utils import generator
from utils import unpacker
from variables import SAVE, DATA, METHODS, SENTENCES, FILTERED, EPOCHS, FEATURES, WINDOW

if __name__ == '__main__':
    methods = unpacker.unpackMethods(METHODS)
    generator.generateTextSet(methods, SENTENCES)
    methods = filter.applyFiltersForMethods(methods)
    generator.generateTextSet(methods, FILTERED)
    generator.generateEmbeddings(SAVE, DATA, EPOCHS, FEATURES, WINDOW)
