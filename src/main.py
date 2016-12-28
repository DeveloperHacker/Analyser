import os

import filters
import generators
import unpackers

if __name__ == '__main__':
    methods = unpackers.unpackJavaDoc(os.getcwd() + "/../resources/packJavaDocs.xml")
    sentences = generators.generateTextSet(methods, os.getcwd() + "/../resources/sentences.txt")
    methods = filters.applyFiltersForMethods(methods)
    sentences = generators.generateTextSet(methods, os.getcwd() + "/../resources/sentencesFiltered.txt")
