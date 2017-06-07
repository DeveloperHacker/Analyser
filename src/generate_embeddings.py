import logging
import sys
from multiprocessing.pool import Pool

from constants.paths import FILTERED, GENERATOR_RAW_METHODS, GENERATOR_LOG
from utils.filters import apply_filters, NEXT
from utils.method import Method
from utils.unpacker import unpack_methods
from utils.wrapper import trace
from word2vec.word2vec import generate, cluster


def join_java_doc(method: Method) -> Method:
    java_doc = method.java_doc
    method.java_doc = {
        "head": java_doc.head,
        "param": (" %s " % NEXT).join(java_doc.params),
        "variable": (" %s " % NEXT).join(java_doc.variables),
        "return": (" %s " % NEXT).join(java_doc.results),
        "see": (" %s " % NEXT).join(java_doc.sees),
        "throw": (" %s " % NEXT).join(java_doc.throws),
    }
    return method


def extract_docs(method: Method):
    java_doc = method.java_doc
    result = (text.strip() for label, text in java_doc.items())
    result = "\n".join(text for text in result if len(text) > 0)
    return result


def apply(method: Method):
    method = apply_filters(method)
    method = join_java_doc(method)
    doc = extract_docs(method)
    return doc


@trace
def prepare():
    methods = unpack_methods(GENERATOR_RAW_METHODS)
    methods = (method for method in methods if not method.java_doc.empty())
    with Pool() as pool:
        docs = pool.map(apply, methods)
    with open(FILTERED, "w") as file:
        file.write("\n".join(docs))


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, filename=GENERATOR_LOG)
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    prepare()
    generate()
    cluster()
