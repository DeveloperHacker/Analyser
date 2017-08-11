import itertools
import random
import re
from multiprocessing.pool import Pool
from typing import Iterable, List, Any, Dict, Tuple

import numpy as np
from contracts import Parser, Tokens, Types, Decompiler
from contracts.BfsGuide import BfsGuide
from contracts.DfsGuide import DfsGuide
from contracts.Node import Node
from contracts.Token import Token
from contracts.Tree import Tree
from contracts.TreeVisitor import TreeVisitor, TreeGuide
from contracts.Validator import is_param, Validator
from pyparsing import ParseException

from configurations.constants import OUTPUT_TYPE, BATCH_SIZE
from configurations.fields import CONTRACT, JAVA_DOC, DESCRIPTION
from configurations.logger import info_logger
from configurations.tags import PAD, NOP, NEXT, PARTS, SIGNATURE, PARAMETER
from seq2seq import Embeddings
from utils import anonymizers
from utils.wrappers import trace, static


def convert(token: Token) -> str:
    if token.type == Types.STRING:
        return Types.STRING
    if token.type == Types.OPERATOR:
        if token.name == Tokens.GREATER_OR_EQUAL:
            return Tokens.LOWER
        if token.name == Tokens.GREATER:
            return Tokens.LOWER_OR_EQUAL
    if token.type == Types.MARKER:
        name = token.name
        if is_param(name) and int(name[len(Tokens.PARAM) + 1:-1]) > 5:
            name = Tokens.PARAM
        return name
    if token.type == Types.LABEL:
        return None
    if token.type == Types.ROOT:
        return None
    return token.name


def convert_to(tree: Tree, height: int) -> List[Tuple[List[str], Dict[int, List[str]]]]:
    @static(trees={1: Node(Token(NOP, Types.MARKER))})
    def empty(height: int) -> Node:
        assert height > 0
        if height not in empty.trees:
            node = empty(height - 1)
            token = empty.trees[1].token
            node = Node(token, node, node)
            empty.trees[height] = node
        return empty.trees[height]

    class Equalizer(TreeVisitor):
        def __init__(self):
            super().__init__(DfsGuide())
            self.tree = None

        def result(self) -> Tree:
            return self.tree

        def visit_tree(self, tree: Tree):
            self.tree = tree

        def visit_node_end(self, depth: int, node: Node, parent: Node):
            diff = height - depth
            if node.leaf() and diff > 0:
                node.children.append(empty(diff))
                node.children.append(empty(diff))

    class Compiler(TreeVisitor):
        def __init__(self, guide: TreeGuide):
            super().__init__(guide)
            self.tokens = None
            self.strings = None

        def result(self) -> Tuple[List[str], Dict[int, List[str]]]:
            return self.tokens, self.strings

        def visit_tree(self, tree: Tree):
            self.tokens = []
            self.strings = {}

        def visit_string(self, depth: int, node: Node, parent: Node):
            index = len(self.tokens)
            string = node.token.name[1:-1].split()
            self.strings[index] = string

        def visit_node_end(self, depth: int, node: Node, parent: Node):
            self.tokens.append(convert(node.token))

    Validator().accept(tree)
    equalizer = Equalizer()
    tree = equalizer.accept(tree)
    forest = [Tree(child.children[0]) for child in tree.root.children]
    if OUTPUT_TYPE in ("tree", "bfs_sequence"):
        compiler = Compiler(BfsGuide())
    elif OUTPUT_TYPE == "dfs_sequence":
        compiler = Compiler(DfsGuide())
    result = [compiler.accept(tree) for tree in forest]
    return result


def convert_from(contract: List[Tuple[List[str], Dict[int, List[str]]]]) -> Tree:
    names = [Tokens.ROOT]
    for i in range(max(len(tokens), len(strings))):
        for j, token in enumerate(tokens[i]):
            if token == Types.STRING:
                string = strings[i][j]
                token = "\"%s\"" % " ".join(string)
            names.append(token)
    tokens = Decompiler.typing(names)
    if OUTPUT_TYPE in ("tree", "bfs_sequence"):
        tree = Decompiler.bfs(tokens)
    elif OUTPUT_TYPE == "dfs_sequence":
        tree = Decompiler.dfs(tokens)
    return tree


def empty(method):
    for label, text in method[JAVA_DOC].items():
        if len(text) > 0:
            return False
    return True


def join_java_doc(method):
    java_doc = {label: " ".join(text) for label, text in method[JAVA_DOC].items()}
    method[JAVA_DOC] = java_doc
    return method


def append_param_delimiter(method):
    parameters = method[JAVA_DOC][PARAMETER]
    parameters = [re.sub(r"(%s\s[^\s]+)" % PARAMETER, r"\1:", text) for text in parameters]
    method[JAVA_DOC][PARAMETER] = parameters
    return method


def append_signature(method):
    description = method[DESCRIPTION]
    del method[DESCRIPTION]
    name = description["name"]
    owner = description["owner"]
    result = description["type"]
    parameters = description["parameters"]
    parameters = [(parameter["name"], parameter["type"]) for parameter in parameters]
    parameters = ", ".join("%s: %s" % parameter for parameter in parameters)
    signature = "%s %s : %s(%s) %s" % (SIGNATURE, owner, name, parameters, result)
    method[JAVA_DOC][SIGNATURE] = [signature]
    return method


def apply_anonymizers(method):
    java_doc = {label: anonymizers.apply(text) for label, text in method[JAVA_DOC].items()}
    method[JAVA_DOC] = java_doc
    return method


def one_line_doc(method):
    java_doc = (method[JAVA_DOC][label].strip() for label in PARTS)
    java_doc = (" %s " % NEXT).join(text for text in java_doc if len(text) > 0)
    method[JAVA_DOC] = java_doc
    return method


def standardify_contract(method):
    class Mapper(TreeVisitor):
        def __init__(self, map_function):
            super().__init__(DfsGuide())
            self.map = map_function

        def visit_node_end(self, depth: int, node: Node, parent: Node):
            children = [self.map(child, parent) for child in node.children]
            node.children = children

    def swap(node: Node, _: Node) -> Node:
        if node.token.type == Types.OPERATOR and node.token.name in (Tokens.GREATER, Tokens.GREATER_OR_EQUAL):
            left = node.children[0]
            right = node.children[1]
            token = Token(convert(node.token), Types.OPERATOR)
            node = Node(token, right, left)
        return node

    def reduce(node: Node, _: Node) -> Node:
        if node.token == Tokens.NOT_EQUAL:
            assert node.children is not None
            assert len(node.children) == 2
            left = node.children[0]
            right = node.children[1]
            if left.token == Tokens.FALSE:
                node = right
            elif right.token == Tokens.FALSE:
                node = left
            elif left.token == Tokens.TRUE:
                node.token = Tokens.EQUAL
                left.token = Tokens.FALSE
                node.children = [right, left]
            elif right.token == Tokens.TRUE:
                node.token = Tokens.EQUAL
                right.token = Tokens.FALSE
        if node.token == Tokens.EQUAL:
            assert node.children is not None
            assert len(node.children) == 2
            left = node.children[0]
            right = node.children[1]
            if left.token == Tokens.TRUE:
                node = right
            elif right.token == Tokens.TRUE:
                node = left
            elif left.token == Tokens.FALSE:
                node.children = [right, left]
        return node

    def expand(node: Node, parent: Node) -> Node:
        children = (Tokens.RESULT, Tokens.GETATTR)
        token = node.token
        cond1 = token.name in children or is_param(token.name) or token.type == Types.STRING
        cond2 = parent is not None and parent.token in (Tokens.ROOT, Tokens.FOLLOW)
        if cond1 and cond2:
            node = Node(Tokens.EQUAL, node, Node(Tokens.TRUE))
        return node

    tree = method[CONTRACT]
    Validator().accept(tree)
    Mapper(swap).accept(tree)
    Mapper(reduce).accept(tree)
    Mapper(expand).accept(tree)
    method[CONTRACT] = tree
    return method


def batching(methods: Iterable[dict]):
    def chunks(iterable: Iterable[Any], block_size: int):
        result = []
        for element in iterable:
            result.append(element)
            if len(result) == block_size:
                yield result
                result = []
        if len(result) > 0:
            yield result

    methods = list(methods)
    random.shuffle(methods)
    return (chunk for chunk in chunks(methods, BATCH_SIZE) if len(chunk) == BATCH_SIZE)


def filter_contract_text(method):
    class StringFiltrator(TreeVisitor):
        def __init__(self):
            super().__init__(DfsGuide())

        def visit_string_end(self, depth: int, node: Node, parent: Node):
            name = node.token.name
            quote = name[0]
            name = anonymizers.apply(name[1:-1])
            name = quote + name + quote
            node.token = Token(name, node.token.type)

    tree = method[CONTRACT]
    filtrator = StringFiltrator()
    filtrator.accept(tree)
    return method


def parse_contract(method):
    try:
        raw_code = method[CONTRACT]
        code = "\n".join(raw_code)
        tree = Parser.parse(code)
        method[CONTRACT] = tree
    except ParseException as ex:
        for i, line in enumerate(raw_code):
            info_logger.info(line)
            if i + 1 == ex.lineno:
                info_logger.info("~" * ex.col + "^")
        raise ex
    except Exception as ex:
        for line in raw_code:
            info_logger.info(line)
        raise ex
    return method


def index_contract(method):
    tree = method[CONTRACT]
    tokens, strings = convert_to(tree)
    result = [
        ([Embeddings.tokens().get_index(token) for token in tokens[i]],
         {idx: [Embeddings.words().get_index(word) for word in string] for idx, string in strings[i].items()})
        for i in range(len(tokens))
    ]
    method[CONTRACT] = result
    return method


def index_java_doc(method):
    java_doc = (word.strip() for word in method[JAVA_DOC].split(" "))
    java_doc = tuple(Embeddings.words().get_index(word) for word in java_doc)
    method[JAVA_DOC] = java_doc
    return method


def build_batch(methods: List[dict]):
    pad = Embeddings.words().get_index(PAD)
    nop = Embeddings.tokens().get_index(NOP)

    inputs = (method[JAVA_DOC] for method in methods)
    inputs = ((word.strip() for word in input.split(" ")) for input in inputs)
    inputs = ((word for word in input if len(word) > 0) for input in inputs)
    inputs = [[Embeddings.words().get_index(word) for word in input] for input in inputs]
    inputs_length = [len(input) for input in inputs]
    inputs_steps = max(inputs_length)
    inputs = [input + [pad] * (inputs_steps + 1 - len(input)) for input in inputs]
    inputs = np.asarray(inputs)
    inputs_length = np.asarray(inputs_length)

    contracts = [method[CONTRACT] for method in methods]
    height = max(contract.height() for contract in contracts)
    num_conditions = max(len(contract.root.children) for contract in contracts)
    contracts = [convert_to(contract, height) for contract in contracts]
    height -= 2
    sequence_length = 2 ** height - 1

    chain = itertools.chain
    mapper = lambda x: x[1].values()
    lengths = chain((0,), map(len, chain(*map(mapper, chain(*contracts)))))
    string_length = max(lengths) + 1

    strings = []
    tokens = []
    for contract in contracts:
        _strings = np.tile(-1, [num_conditions, sequence_length, string_length])
        _tokens = np.tile(nop, [num_conditions, sequence_length])
        for i, (raw_tokens, raw_strings) in enumerate(contract):
            raw_tokens = [Embeddings.tokens().get_index(token) for token in raw_tokens]
            _tokens[i][:len(raw_tokens)] = raw_tokens
            for idx, raw_string in raw_strings.items():
                raw_string = [Embeddings.words().get_index(word) for word in raw_string]
                _strings[i][idx][:len(raw_string)] = raw_string
                _strings[i][idx][len(raw_string)] = pad
        strings.append(_strings)
        tokens.append(_tokens)
    tokens = np.asarray(tokens)
    strings = np.asarray(strings)

    inputs = (inputs, inputs_length)
    outputs = (tokens, strings)
    parameters = (num_conditions, sequence_length, string_length, height)
    return inputs, outputs, parameters


@trace("PREPARE JAVA-DOC")
def java_doc(methods) -> List[dict]:
    with Pool() as pool:
        methods = (method for method in methods if not empty(method))
        methods = pool.map(append_param_delimiter, methods)
        methods = pool.map(append_signature, methods)
        methods = pool.map(join_java_doc, methods)
        methods = pool.map(apply_anonymizers, methods)
        methods = (method for method in methods if not empty(method))
        methods = pool.map(one_line_doc, methods)
    return list(methods)


@trace("PREPARE CONTRACT")
def contract(methods) -> List[dict]:
    with Pool() as pool:
        methods = pool.map(parse_contract, methods)
        methods = (method for method in methods if method[CONTRACT].height() > 1)
        methods = pool.map(standardify_contract, methods)
        methods = pool.map(filter_contract_text, methods)
    return list(methods)


@trace("PREPARE BATCHES")
def batches(methods) -> list:
    with Pool() as pool:
        batches = batching(methods)
        batches = pool.map(build_batch, batches)
        random.shuffle(batches)
    return list(batches)
