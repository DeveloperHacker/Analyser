import json
import random
import re
from typing import Iterable, List, Any, Dict, Tuple

import numpy as np
from contracts import Parser, Tokens, Types
from contracts.BfsGuide import BfsGuide
from contracts.DfsGuide import DfsGuide
from contracts.Node import Node
from contracts.Token import Token
from contracts.Tree import Tree
from contracts.TreeVisitor import TreeVisitor, TreeGuide
from contracts.Validator import is_param, Validator

from analyser import Embeddings
from contants import PAD, NOP, NEXT, PARTS, SIGNATURE, PARAMETER, CONTRACT, JAVA_DOC, DESCRIPTION
from utils import anonymizers
from utils.wrappers import static

MAX_PARAM = 5


def convert(token: Token) -> Token:
    if token.type == Types.STRING:
        return Token(Types.STRING, Types.STRING)
    if token.type == Types.OPERATOR:
        if token.name == Tokens.GREATER_OR_EQUAL:
            return Token(Tokens.LOWER, Types.OPERATOR)
        if token.name == Tokens.GREATER:
            return Token(Tokens.LOWER_OR_EQUAL, Types.OPERATOR)
        return token
    if token.type == Types.MARKER:
        name = token.name
        index = name[len(Tokens.PARAM) + 1:-1]
        if is_param(name) and int(index) > MAX_PARAM:
            return Token(Tokens.PARAM, Types.MARKER)
        return Token(name, Types.MARKER)
    return None


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
            token = convert(node.token)
            if token is not None:
                self.tokens.append(token.name)

    Validator().accept(tree)
    equalizer = Equalizer()
    tree = equalizer.accept(tree)
    forest = [Tree(child.children[0]) for child in tree.root.children]
    compiler = Compiler(BfsGuide())
    result = [compiler.accept(tree) for tree in forest]
    return result


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
            children = [self.map(child, node) for child in node.children]
            node.children = children

    def swap(node: Node, _: Node) -> Node:
        comparators = (Tokens.GREATER, Tokens.GREATER_OR_EQUAL)
        if node.token.type == Types.OPERATOR and node.token.name in comparators:
            node.children = node.children[::-1]
            node.token = convert(node.token)
        return node

    def reduce(node: Node, _: Node) -> Node:
        if node.token.name == Tokens.NOT_EQUAL:
            assert node.children is not None
            assert len(node.children) == 2
            left = node.children[0]
            right = node.children[1]
            if left.token.name == Tokens.FALSE:
                node = right
            elif right.token.name == Tokens.FALSE:
                node = left
            elif left.token.name == Tokens.TRUE:
                node.token = Token(Tokens.EQUAL, Types.OPERATOR)
                left.token = Token(Tokens.FALSE, Types.MARKER)
                node.children = [right, left]
            elif right.token.name == Tokens.TRUE:
                node.token = Token(Tokens.EQUAL, Types.OPERATOR)
                right.token = Token(Tokens.FALSE, Types.MARKER)
        if node.token.name == Tokens.EQUAL:
            assert node.children is not None
            assert len(node.children) == 2
            left = node.children[0]
            right = node.children[1]
            if left.token.name == Tokens.TRUE:
                node = right
            elif right.token.name == Tokens.TRUE:
                node = left
            elif left.token.name == Tokens.FALSE:
                node.children = node.children[::-1]
        return node

    def expand(node: Node, parent: Node) -> Node:
        cond = node.token.type == Types.MARKER and parent is not None
        cond1 = cond and parent.token.name in (Tokens.FOLLOW, Tokens.AND, Tokens.OR)
        cond2 = cond and parent.token.type == Types.LABEL
        if cond1 or cond2:
            token = Token(Tokens.EQUAL, Types.OPERATOR)
            right = Node(Token(Tokens.TRUE, Types.OPERATOR))
            node = Node(token, node, right)
        return node

    tree = method[CONTRACT]
    Validator().accept(tree)
    Mapper(swap).accept(tree)
    Mapper(reduce).accept(tree)
    Mapper(expand).accept(tree)
    method[CONTRACT] = tree
    return method


def batching(methods: Iterable[dict], batch_size: int):
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
    return (chunk for chunk in chunks(methods, batch_size) if len(chunk) == batch_size)


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
    raw_code = method[CONTRACT]
    code = "\n".join(raw_code)
    tree = Parser.parse(code)
    method[CONTRACT] = tree
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

    strings_lengths = (len(string)
                       for contract in contracts
                       for tokens, strings in contract
                       for string in strings.values())
    string_length = max(strings_lengths, default=0) + 1

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
                _strings[i][idx][len(raw_string):] = [pad] * (string_length - len(raw_string))
        strings.append(_strings)
        tokens.append(_tokens)
    tokens = np.asarray(tokens)
    strings = np.asarray(strings)

    inputs = (inputs, inputs_length)
    outputs = (tokens, strings)
    parameters = (num_conditions, sequence_length, string_length, height)
    return inputs, outputs, parameters


def load(path: str) -> Iterable[dict]:
    with open(path) as file:
        strings = (line for line in file if not line.strip().startswith("//"))
        methods = json.loads("\n".join(strings))
    return methods


def java_doc(methods) -> Iterable[dict]:
    methods = (method for method in methods if not empty(method))
    methods = (append_param_delimiter(method) for method in methods)
    methods = (append_signature(method) for method in methods)
    methods = (join_java_doc(method) for method in methods)
    methods = (apply_anonymizers(method) for method in methods)
    methods = (method for method in methods if not empty(method))
    methods = (one_line_doc(method) for method in methods)
    return methods


def contract(methods) -> Iterable[dict]:
    methods = (parse_contract(method) for method in methods)
    methods = (method for method in methods if method[CONTRACT].height() > 1)
    methods = (standardify_contract(method) for method in methods)
    methods = (filter_contract_text(method) for method in methods)
    return methods


def batches(methods, batch_size) -> list:
    batches = batching(methods, batch_size)
    batches = [build_batch(method) for method in batches]
    random.shuffle(batches)
    return batches
