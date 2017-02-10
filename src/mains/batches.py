import _pickle as pickle

from utils import constructor
from utils import filter
from utils import unpacker
from variables import METHODS, BATCHES, BATCH_SIZE

if __name__ == '__main__':
    print("unpack")
    methods = unpacker.unpackMethods(METHODS)
    print("filter")
    methods = filter.applyFiltersForMethods(methods)
    print("batches")
    methods = [method for method in methods if not method.javaDoc.empty()]
    batches = constructor.batching(methods, BATCH_SIZE)
    print("pickle")
    with open(BATCHES, "wb") as file:
        pickle.dump(batches, file)
