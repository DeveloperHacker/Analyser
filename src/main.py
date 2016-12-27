import os

import filters
import generators
import unpackers

if __name__ == '__main__':
    print(filters.filterTags("sajdkh aksdjhask hdaks kjhd <asdasd asd asd> ka adskjk hs kda </asdasd asd asd> ksdh aks"))

    methods = unpackers.unpackJavaDoc(os.getcwd() + "/../resources/packJavaDocs.xml")
    generators.generateVocabulary(methods, os.getcwd() + "/../resources/vocabulary.txt")
