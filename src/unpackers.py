import xml.etree.ElementTree
from xml.etree.ElementTree import ElementTree

from parts import *


class Tags:
    methods = "methods"
    method = "method"
    javaDoc = "javaDoc"
    head = "head"
    param = "param"
    result = "return"
    see = "see"
    throws = "throws"
    description = "description"
    name = "name"
    type = "type"
    parameters = "parameters"
    owner = "owner"


def unpackJavaDoc(absoluteFileName: str) -> list:
    parser = xml.etree.ElementTree.parse(absoluteFileName).getroot()  # type: ElementTree
    methods = []
    for methodTag in parser.findall(Tags.method):
        method = Method()
        for javaDocTag in methodTag.findall(Tags.javaDoc):
            javaDoc = JavaDoc()
            for headTag in javaDocTag.findall(Tags.head):
                javaDoc.head = headTag.text
            for paramTag in javaDocTag.findall(Tags.param):
                javaDoc.params.append(paramTag.text)
            for result in javaDocTag.findall(Tags.result):
                javaDoc.results.append(result.text)
            for see in javaDocTag.findall(Tags.see):
                javaDoc.sees.append(see.text)
            for throw in javaDocTag.findall(Tags.throws):
                javaDoc.throws.append(throw.text)
            method.javaDoc = javaDoc
        for descriptionTag in methodTag.findall(Tags.description):
            for nameTag in descriptionTag.findall(Tags.name):
                method.name = nameTag.text
            for typeTAg in descriptionTag.findall(Tags.type):
                method.type = Type(typeTAg.text)
            for paramsTag in descriptionTag.findall(Tags.parameters):
                for paramTag in paramsTag.findall(Tags.param):
                    parameter = Parameter()
                    for nameTag in paramTag.findall(Tags.name):
                        parameter.name = nameTag.text
                    for typeTAg in paramTag.findall(Tags.type):
                        parameter.type = Type(typeTAg.text)
                    method.params.append(parameter)
            for owner in descriptionTag.findall(Tags.owner):
                method.owner = Type(owner.text)
        methods.append(method)
    return methods
