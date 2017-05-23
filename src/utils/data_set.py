from multiprocessing.pool import Pool

from contracts.nodes.StringNode import StringNode
from contracts.nodes.WordNode import WordNode
from contracts.parser.Parser import Parser
from contracts.visitors.AstCompiler import AstCompiler
from contracts.visitors.AstVisitor import AstVisitor

from utils import unpacker, filters, dumper
from utils.method import Method
from utils.wrapper import trace
from variables.embeddings import WordEmbeddings, TokenEmbeddings
from variables.paths import PREPARED_METHODS
from variables.tags import NEXT


class StringFiltrator(AstVisitor):
    def __init__(self, method: Method):
        super().__init__()
        self.method = method

    def _visit_string(self, node: StringNode):
        string = " ".join(word.instance for word in node.children)
        string = filters.applyFiltersForString(string, self.method.get_param_names())
        node.children = [WordNode(word) for word in string.split(" ")]


@trace
def prepare_data_set():
    methods = unpacker.unpack_methods()
    with Pool() as pool:
        methods = pool.map(filters.applyFiltersForMethod, methods)
        methods = (method for method in methods if not method.java_doc.empty())
        methods = pool.map(join_java_doc, methods)
        methods = pool.map(index_java_doc, methods)
        methods = pool.map(parse_contract, methods)
        methods = (method for method in methods if len(method.contract) > 1)
        methods = pool.map(filter_contract_text, methods)
        methods = pool.map(index_contract, methods)
    print(len(methods))
    for method in methods:
        print(", ".join(str(idx) for idx in method.java_doc))
        print(method.description)
        print(", ".join(str(idx) for idx in method.contract))
        print()
    dumper.dump(methods, PREPARED_METHODS)


def filter_contract_text(method: Method) -> Method:
    tree = Parser.tree(method.contract)
    filtrator = StringFiltrator(method)
    filtrator.accept(tree)
    compiler = AstCompiler()
    compiler.accept(tree)
    method.contract = compiler.instructions
    return method


def parse_contract(method: Method) -> Method:
    method.contract = Parser.parse(method.contract.code)
    return method


def index_contract(method: Method) -> Method:
    result = []
    for instruction in method.contract:
        token_index = TokenEmbeddings.get_index(instruction.token.name)
        word_index = WordEmbeddings.get_index(instruction.word)
        result.append((token_index, word_index))
    method.contract = result
    return method


def index_java_doc(method: Method) -> Method:
    result = []
    for label, text in method.java_doc:
        split = text.split(" ")
        indexes = tuple(WordEmbeddings.get_index(word) for word in split)
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
