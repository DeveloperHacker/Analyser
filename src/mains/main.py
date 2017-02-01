from utils import unpacker
from utils import filter
from mains.variables import METHODS

if __name__ == '__main__':
    methods = unpacker.unpackMethods(METHODS)
    # for method in methods:
    #     if not method.javaDoc.empty():
    #         print(method.javaDoc)
    #         print(method.description)
    #         print()
    methods = filter.applyFiltersForMethods(methods)
    for method in methods:
        if not method.javaDoc.empty():
            print(method.javaDoc)
            print(method.description)
            print()
