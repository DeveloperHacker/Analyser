import re
from collections import namedtuple
from random import Random
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
from contants import PAD, NOP, NEXT, PARTS, SIGNATURE, PARAMETER, CONTRACT, JAVA_DOC, DESCRIPTION, UNDEFINED
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
    if token.type == Types.LABEL:
        return token
    return None


@static(trees={1: Node(Token(NOP, Types.MARKER))})
def empty_tree(height: int) -> Node:
    assert height > 0
    if height not in empty_tree.trees:
        node = empty_tree(height - 1)
        token = empty_tree.trees[1].token
        node = Node(token, node, node)
        empty_tree.trees[height] = node
    return empty_tree.trees[height]


class Equalizer(TreeVisitor):
    def __init__(self, height: int):
        super().__init__(DfsGuide())
        self.tree = None
        self.height = height

    def result(self) -> Tree:
        return self.tree

    def visit_tree(self, tree: Tree):
        self.tree = tree

    def visit_node_end(self, depth: int, node: Node, parent: Node):
        diff = self.height - depth
        if node.leaf() and diff > 0:
            node.children.append(empty_tree(diff))
            node.children.append(empty_tree(diff))


class Compiler(TreeVisitor):
    def __init__(self, guide: TreeGuide):
        super().__init__(guide)
        self.label = None
        self.tokens = None
        self.strings = None

    def result(self) -> Tuple[List[str], Dict[int, List[str]]]:
        return self.label, self.tokens, self.strings

    def visit_tree(self, tree: Tree):
        self.label = UNDEFINED
        self.tokens = []
        self.strings = {}

    def visit_label(self, depth: int, node: Node, parent: Node):
        self.label = node.token.name

    def visit_string(self, depth: int, node: Node, parent: Node):
        index = len(self.tokens)
        string = node.token.name[1:-1].split()
        self.strings[index] = string

    def visit_node_end(self, depth: int, node: Node, parent: Node):
        token = convert(node.token)
        if token is not None and token.type != Types.LABEL:
            self.tokens.append(token.name)


def convert_to(tree: Tree, height: int, filling: bool, flatten_type: str):
    Validator().accept(tree)
    if filling:
        equalizer = Equalizer(height)
        tree = equalizer.accept(tree)
    forest = [Tree(child) for child in tree.root.children]
    if flatten_type == "bfs":
        compiler = Compiler(BfsGuide())
    elif flatten_type == "dfs":
        compiler = Compiler(DfsGuide())
    else:
        raise ValueError("Flatten type '%s' hasn't recognised" % flatten_type)
    result = [compiler.accept(tree) for tree in forest]
    return result


def is_empty(method):
    for label, text in method[JAVA_DOC].items():
        if len(text) > 0:
            return False
    return True


def join_java_doc(method):
    java_doc = {label: " ".join(text) for label, text in method[JAVA_DOC].items()}
    method[JAVA_DOC] = java_doc
    return method


def append_param_delimiter(method):
    if PARAMETER in method[JAVA_DOC]:
        parameters = method[JAVA_DOC][PARAMETER]
        parameters = [re.sub(r"(%s\s[^\s]+)" % PARAMETER, r"\1:", text) for text in parameters]
        method[JAVA_DOC][PARAMETER] = parameters
    return method


def append_signature(method):
    description = method[DESCRIPTION]
    # method[JAVA_DOC][SIGNATURE] = [description[FLAT]]
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
    java_doc = (method[JAVA_DOC].get(label, "").strip() for label in PARTS)
    java_doc = (" %s " % NEXT).join(text for text in java_doc if len(text) > 0)
    method[JAVA_DOC] = java_doc
    return method


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


def standardify_contract(method):
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


def build_batch(methods: List[dict], filling, flatten_type):
    undefined = Embeddings.labels().get_index(UNDEFINED)
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
    labels_length = max(len(contract.root.children) for contract in contracts)
    contracts = [convert_to(contract, height, filling, flatten_type) for contract in contracts]
    height -= 2
    tokens_length = 2 ** height - 1

    strings_lengths = (
        len(string)
        for contract in contracts
        for label, tokens, strings in contract
        for string in strings.values())
    strings_length = max(strings_lengths, default=0) + 1

    labels_targets, tokens_targets, strings_targets = [], [], []
    for contract in contracts:
        strings = np.tile(-1, [labels_length, tokens_length, strings_length])
        tokens = np.tile(nop, [labels_length, tokens_length])
        labels = np.tile(undefined, [labels_length])
        for i, (raw_label, raw_tokens, raw_strings) in enumerate(contract):
            labels[i] = Embeddings.labels().get_index(raw_label)
            raw_tokens = [Embeddings.tokens().get_index(token) for token in raw_tokens]
            tokens[i][:len(raw_tokens)] = raw_tokens
            for idx, raw_string in raw_strings.items():
                raw_string = [Embeddings.words().get_index(word) for word in raw_string]
                strings[i][idx][:len(raw_string)] = raw_string
                strings[i][idx][len(raw_string):] = [pad] * (strings_length - len(raw_string))
        labels_targets.append(labels)
        tokens_targets.append(tokens)
        strings_targets.append(strings)
    labels_targets = np.asarray(labels_targets)
    tokens_targets = np.asarray(tokens_targets)
    strings_targets = np.asarray(strings_targets)

    labels = labels_targets, labels_length
    tokens = tokens_targets, tokens_length
    strings = strings_targets, strings_length
    return (inputs, inputs_length), labels, tokens, strings


def java_doc(methods) -> Iterable[dict]:
    methods = (method for method in methods if not is_empty(method))
    methods = (append_param_delimiter(method) for method in methods)
    methods = (append_signature(method) for method in methods)
    methods = (join_java_doc(method) for method in methods)
    methods = (apply_anonymizers(method) for method in methods)
    methods = (method for method in methods if not is_empty(method))
    methods = (one_line_doc(method) for method in methods)
    return methods


def contract(methods) -> Iterable[dict]:
    methods = (parse_contract(method) for method in methods)
    methods = (method for method in methods if method[CONTRACT].height() > 1)
    methods = (standardify_contract(method) for method in methods)
    methods = (filter_contract_text(method) for method in methods)
    return methods


def batches(methods, batch_size, filling, flatten_type) -> list:
    batches = batching(methods, batch_size)
    batches = [build_batch(method, filling, flatten_type) for method in batches]
    return batches


DataSet = namedtuple("DataSet", ("train", "validation", "test"))


def part(batches, train, validation, test, random: Random) -> DataSet:
    random.shuffle(batches)
    data_set_length = len(batches)
    test_set_length = int(data_set_length * test)
    not_allocated = data_set_length - test_set_length
    train_set_length = min(not_allocated, int(data_set_length * train))
    not_allocated = data_set_length - test_set_length - train_set_length
    validation_set_length = min(not_allocated, int(data_set_length * validation))
    test_set = batches[-test_set_length:]
    batches = batches[:-test_set_length]
    random.shuffle(batches)
    train_set = batches[-train_set_length:]
    batches = batches[:-train_set_length]
    validation_set = batches[-validation_set_length:]
    return DataSet(train_set, validation_set, test_set)
