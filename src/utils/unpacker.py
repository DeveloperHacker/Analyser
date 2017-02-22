import xml.etree.ElementTree
from xml.etree.ElementTree import ElementTree

import numpy
import tensorflow

from utils import dumper
from utils.method import *
from variables import EMB_STORAGE, EMB_MODEL, METHODS


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
    contract = "contract"
    enter = "enter"
    enters = "enters"
    exit = "exit"
    exits = "exits"
    exitId = "exitId"
    exitIds = "exitIds"


def unpackMethods() -> list:
    parser = xml.etree.ElementTree.parse(METHODS).getroot()  # type: ElementTree
    methods = []
    for methodTag in parser.findall(Tags.method):
        javaDoc = JavaDoc()
        for javaDocTag in methodTag.findall(Tags.javaDoc):
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
        description = Description()
        for descriptionTag in methodTag.findall(Tags.description):
            for nameTag in descriptionTag.findall(Tags.name):
                description.name = nameTag.text
            for typeTAg in descriptionTag.findall(Tags.type):
                description.type = Type(typeTAg.text)
            for paramsTag in descriptionTag.findall(Tags.parameters):
                for paramTag in paramsTag.findall(Tags.param):
                    parameter = Parameter()
                    for nameTag in paramTag.findall(Tags.name):
                        parameter.name = nameTag.text
                    for typeTAg in paramTag.findall(Tags.type):
                        parameter.type = Type(typeTAg.text)
                        description.params.append(parameter)
            for owner in descriptionTag.findall(Tags.owner):
                description.owner = Type(owner.text)
        contract = Contract()
        for contractTag in methodTag.findall(Tags.contract):
            for entersTag in contractTag.findall(Tags.enters):
                for enterTag in entersTag.findall(Tags.enter):
                    contract.enters.append(enterTag.text)
            for exitsTag in contractTag.findall(Tags.exits):
                for exitTag in exitsTag.findall(Tags.exit):
                    contract.exits.append(exitTag.text)
            for exitIdsTag in contractTag.findall(Tags.exitIds):
                exitIds = {"id": None, "exits": []}
                for exitIdTag in exitIdsTag.findall(Tags.exitId):
                    exitIds["id"] = int(exitIdTag.text)
                for exitsTag in exitIdsTag.findall(Tags.exits):
                    for exitTag in exitsTag.findall(Tags.exit):
                        exitIds["exits"].append(exitTag.text)
                if exitIds["id"] is not None:
                    contract.exitIds.append(exitIds)
        method = Method()
        method.description = description
        method.javaDoc = javaDoc
        method.contract = contract
        methods.append(method)
    return methods


def unpackEmbeddings():
    storage = dumper.load(EMB_STORAGE)

    emb_dim = storage.options.emb_dim
    vocab_size = storage.options.vocab_size

    w_in = tensorflow.Variable(tensorflow.zeros([vocab_size, emb_dim]), name="w_in")
    w_out = tensorflow.Variable(tensorflow.zeros([vocab_size, emb_dim]), name="w_out")
    global_step = tensorflow.Variable(0, name="global_step")

    saver = tensorflow.train.Saver()

    with tensorflow.Session() as sess:
        saver.restore(sess, EMB_MODEL)
        emb = w_in.eval(sess)  # type: numpy.multiarray.ndarray
        embeddings = {word.decode("utf8", errors='replace'): emb[i] for word, i in storage.word2id.items()}

    return embeddings
