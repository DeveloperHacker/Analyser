from utils import constructor
from utils import filter
from utils import unpacker
from variables import METHODS

if __name__ == '__main__':
    print("unpack")
    methods = unpacker.unpackMethods(METHODS)
    print("filter")
    methods = filter.applyFiltersForMethods(methods)
    print("construct")
    rnn = constructor.constructRNNNet(methods)
