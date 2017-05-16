from multiprocessing.pool import Pool

from contract_parser.Parser import Parser

from utils import unpacker, filters, dumper
from utils.method import Method
from variables.embeddings import Embeddings
from variables.paths import PREPARED_METHODS
from variables.tags import NEXT


def prepare_data_set():
    methods = unpacker.unpack_methods()
    with Pool() as pool:
        methods = pool.map(filters.applyFiltersForMethod, methods)
        methods = (method for method in methods if not method.java_doc.empty())
        methods = pool.map(join_java_doc, methods)
        methods = pool.map(index_java_doc, methods)
        methods = pool.map(parse_contract, methods)
        methods = pool.map(index_contract, methods)
    dumper.dump(methods, PREPARED_METHODS)


def parse_contract(method: Method) -> Method:
    method.contract = Parser.parse(method.contract)
    return method


def index_contract(method: Method) -> Method:
    pass
    return method


def index_java_doc(method: Method) -> Method:
    result = []
    for label, text in method.java_doc:
        split = text.split(" ")
        indexes = tuple(Embeddings.get_index(word) for word in split)
        result.append((label, indexes))
    method.java_doc = tuple(result)
    return method


def join_java_doc(method: Method) -> Method:
    java_doc = method.java_doc
    method.java_doc = (
        ("head", java_doc.head),
        ("param", (" %s " % NEXT).join(java_doc.params)),
        ("variable", (" %s " % NEXT).join(java_doc.variables)),
        ("return", (" %s " % NEXT).join(java_doc.results)),
        # ("see", (" %s " % NEXT).join(java_doc.sees)),
        # ("throw", (" %s " % NEXT).join(java_doc.throws))
    )
    return method
