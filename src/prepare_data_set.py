import itertools
import os
import random
from typing import List, Any, Iterable, Tuple

import numpy as np
from contracts.guides.AstBfsGuide import AstBfsGuide
from contracts.guides.AstDfsGuide import AstDfsGuide
from contracts.nodes.Ast import Ast
from contracts.nodes.Node import Node
from contracts.nodes.StringNode import StringNode
from contracts.parser import Parser
from contracts.tokens import tokens
from contracts.tokens.tokens import WEAK
from contracts.visitors.AstCompiler import AstCompiler
from contracts.visitors.AstDecompiler import AstDecompiler
from contracts.visitors.AstVisitor import AstVisitor

from config import init
from constants import embeddings
from constants.analyser import BATCH_SIZE
from constants.paths import JODA_TIME_DATA_SET, ANALYSER_METHODS
from constants.tags import PARTS, PAD, NOP
from generate_embeddings import join_java_doc, empty
from utils import filters, dumpers
from utils.Formatter import Formatter
from utils.wrappers import trace


class Statistic:
    class Accountant:
        def __init__(self):
            self._methods_counter = 0
            self._token_counters = {token: 0 for token in itertools.chain(tokens.predicates(), tokens.markers())}
            self._label_counters = {token: 0 for token in tokens.labels()}

        def consider(self, method):
            self._methods_counter += 1
            for label, instructions, strings in method["contract"]:
                for instruction in instructions:
                    token = instruction.token
                    self._token_counters[token.name] += 1
                self._label_counters[label.name] += 1
            return method

        @property
        def tokens(self) -> Iterable[Tuple[str, int]]:
            return self._token_counters.items()

        @property
        def labels(self) -> Iterable[Tuple[str, int]]:
            return self._label_counters.items()

        @property
        def num_tokens(self) -> int:
            return sum(counter for counter in self._token_counters.values())

        @property
        def num_labels(self):
            return sum(counter for counter in self._label_counters.values())

        @property
        def num_methods(self) -> int:
            return self._methods_counter

    def __init__(self):
        self.before = Statistic.Accountant()
        self.after = Statistic.Accountant()

    def show(self):
        formatter = Formatter(("name", "before", "after"), ("s", "d", "d"), (20, 20, 20), (0, 1, 2))
        formatter.print_head()
        formatter.print("methods", self.before.num_methods, self.after.num_methods)
        formatter.print("labels", self.before.num_labels, self.after.num_labels)
        formatter.print("tokens", self.before.num_tokens, self.after.num_tokens)
        formatter.print_delimiter()
        values = []
        for before, after in zip(self.before.labels, self.after.labels):
            assert before[0] == after[0]
            values.append((before[0], before[1], after[1]))
        for name, before, after in sorted(values, key=lambda x: x[1], reverse=True):
            formatter.print(name, before, after)
        formatter.print_delimiter()
        values = []
        for before, after in zip(self.before.tokens, self.after.tokens):
            assert before[0] == after[0]
            values.append((before[0], before[1], after[1]))
        for name, before, after in sorted(values, key=lambda x: x[1], reverse=True):
            formatter.print(name, before, after)
        formatter.print_lower_delimiter()


def standardify_contract(method):
    class AstMapper(AstVisitor):
        def __init__(self, map_function):
            self.tree = None
            self.map = map_function

        def visit_end(self, ast: Ast):
            ast.root = self.map(ast.root)
            self.tree = ast

        def visit_node_end(self, node: Node):
            children = [self.map(child) for child in node.children]
            node.children = children

        def result(self):
            return self.tree

    def reduce(node: Node):
        parent = node.parent
        if node.token == tokens.NOT_EQUAL:
            assert node.children is not None
            assert len(node.children) == 2
            left = node.children[0]
            right = node.children[1]
            if left.token == tokens.FALSE:
                node = right
            elif right.token == tokens.FALSE:
                node = left
            elif left.token == tokens.TRUE:
                node.token = tokens.EQUAL
                left.token = tokens.FALSE
                node.children = [right, left]
            elif right.token == tokens.TRUE:
                node.token = tokens.EQUAL
                right.token = tokens.FALSE
        if node.token == tokens.EQUAL:
            assert node.children is not None
            assert len(node.children) == 2
            left = node.children[0]
            right = node.children[1]
            if left.token == tokens.TRUE:
                node = right
            elif right.token == tokens.TRUE:
                node = left
            elif left.token == tokens.FALSE:
                node.children = [right, left]
        node.parent = parent
        return node

    def expand(node: Node):
        children = (tokens.PARAM, tokens.PARAM_0, tokens.PARAM_1, tokens.PARAM_2, tokens.PARAM_3, tokens.PARAM_4,
                    tokens.RESULT, tokens.STRING, tokens.GET)
        if node.token in children and (node.parent is None or node.parent.token == tokens.FOLLOW):
            node = Node(tokens.EQUAL, (node, Node(tokens.TRUE)), node.parent)
        return node

    reducer = AstDfsGuide(AstMapper(reduce))
    expander = AstDfsGuide(AstMapper(expand))
    compiler = AstDfsGuide(AstCompiler())
    forest = (Parser.parse_tree(*args) for args in method["contract"])
    forest = (reducer.accept(tree) for tree in forest)
    forest = list(expander.accept(tree) for tree in forest)
    decompiler = AstDfsGuide(AstDecompiler())
    for tree in forest:
        print(decompiler.accept(tree))
    method["contract"] = [compiler.accept(tree) for tree in forest]
    return method


def filter_contract(method):
    method["contract"] = [
        (label, instructions, strings)
        for label, instructions, strings in method["contract"]
        if label != WEAK
    ]
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
    class StringFiltrator(AstVisitor):
        def __init__(self, method):
            super().__init__()
            self._tree = None
            self._params = [param["name"] for param in method["description"]["parameters"]]

        def visit(self, ast: Ast):
            self._tree = ast

        def result(self):
            return self._tree

        def visit_string(self, node: StringNode):
            string = " ".join(node.words)
            string = filters.apply_filters(string, self._params)
            node.words = string.split(" ")

    filtrator = AstDfsGuide(StringFiltrator(method))
    output_type = os.environ['OUTPUT_TYPE']
    if output_type in ("tree", "bfs_sequence"):
        compiler = AstBfsGuide(AstCompiler())
    elif output_type == "dfs_sequence":
        compiler = AstDfsGuide(AstCompiler())
    forest = (Parser.parse_tree(*args) for args in method["contract"])
    forest = (filtrator.accept(tree) for tree in forest)
    method["contract"] = [compiler.accept(tree) for tree in forest]
    return method


def parse_contract(method):
    forest = Parser.parse("\n".join(method["contract"]))
    compiler = AstDfsGuide(AstCompiler())
    method["contract"] = [compiler.accept(tree) for tree in forest]
    return method


def index_contract(method):
    result = []
    for label, instructions, strings in method["contract"]:
        label = embeddings.labels().get_index(label.name)
        instructions = [
            embeddings.tokens().get_index(instruction.token.name)
            for instruction in instructions
        ]
        strings = {
            idx: [embeddings.words().get_index(word) for word in string]
            for idx, string in strings.items()
        }
        result.append((label, instructions, strings))
    method["contract"] = result
    return method


def index_java_doc(method):
    result = {}
    for label, text in method["java-doc"].items():
        split = (word.strip() for word in text.split(" "))
        indexes = tuple(embeddings.words().get_index(word) for word in split if len(word) > 0)
        result[label] = indexes
    method["java-doc"] = result
    return method


def build_batch(methods: List[dict]):
    inputs_steps = {label: max([len(method["java-doc"][label]) for method in methods]) for label in PARTS}
    docs = {label: [] for label in PARTS}
    docs_sizes = {label: [] for label in PARTS}
    pad = embeddings.words().get_index(PAD)
    for method in methods:
        for label in PARTS:
            line = list(method["java-doc"][label])
            docs_sizes[label].append(len(line))
            expected = inputs_steps[label] + 1 - len(line)
            line = line + [pad] * expected
            docs[label].append(line)
    for label in PARTS:
        docs[label] = np.transpose(np.asarray(docs[label]), (1, 0))
        docs_sizes[label] = np.asarray(docs_sizes[label])
    num_conditions = []
    sequence_length = []
    strings_lengths = [1]
    for method in methods:
        contract = method["contract"]
        num_conditions.append(len(contract))
        for raw_label, raw_instructions, raw_strings in contract:
            sequence_length.append(len(raw_instructions))
            strings_lengths.extend(len(string) for idx, string in raw_strings.items())
    string_length = max(strings_lengths)
    num_conditions = max(num_conditions)
    sequence_length = max(sequence_length)
    tree_depth = int(np.ceil(np.log2(sequence_length)))
    output_type = os.environ['OUTPUT_TYPE']
    if output_type == "tree":
        sequence_length = 2 ** tree_depth - 1
    strings_mask = []
    strings = []
    tokens = []
    labels = []
    nop = embeddings.tokens().get_index(NOP)
    empty_sequence = [nop] * sequence_length
    empty_string = [0] * string_length
    for method in methods:
        string_mask = [[0] * sequence_length for _ in range(num_conditions)]
        empty_strings = [[empty_string] * sequence_length for _ in range(num_conditions)]
        empty_tokens = [empty_sequence] * num_conditions
        empty_labels = [0] * num_conditions
        for i, (raw_label, raw_instructions, raw_strings) in enumerate(method["contract"]):
            empty_labels[i] = raw_label
            raw_instructions = raw_instructions + [nop] * (sequence_length - len(raw_instructions))
            empty_tokens[i] = raw_instructions
            for idx, raw_string in raw_strings.items():
                raw_string = raw_string + [pad] * (string_length - len(raw_string))
                empty_strings[i][idx] = raw_string
                string_mask[i][idx] = 1
        strings_mask.append(string_mask)
        strings.append(empty_strings)
        tokens.append(empty_tokens)
        labels.append(empty_labels)
    labels = np.asarray(labels)
    tokens = np.asarray(tokens)
    strings = np.asarray(strings)
    strings_mask = np.asarray(strings_mask)
    inputs = (docs, docs_sizes)
    outputs = (labels, tokens, strings, strings_mask)
    parameters = (num_conditions, sequence_length, string_length, tree_depth)
    return inputs, outputs, parameters


# fixme: dirt
def align_data_set(methods: Iterable[dict]):
    # methods = tuple(methods)
    # stats = []
    # for method in methods:
    #     token_counters = {}
    #     for label, instructions, strings in method["contract"]:
    #         for instruction in instructions:
    #             name = instruction.token.name
    #             token_counters[name] = token_counters.get(name, 0) + 1
    #     stats.append(token_counters)
    # token_counters = {}
    # for stat in stats:
    #     for name, number in stat.items():
    #         token_counters[name] = token_counters.get(name, 0) + number
    # max_value = max(token_counters.values())
    # overhead = 10
    # imbalance = 0.1
    # part = 0.4
    # lack = {name: part * max_value - number for name, number in token_counters.items() if number < part * max_value}
    # stats = [[i, stat, 0] for i, stat in enumerate(stats) if any(name in lack for name in stat)]
    # stop = False
    # while not stop:
    #     new_max_value = max(token_counters.values())
    #     if max_value + overhead < new_max_value:
    #         break
    #     min_value = min(number for index, stat, number in stats)
    #     stop = True
    #     for i in range(len(stats)):
    #         index, stat, number = stats[i]
    #         if number * imbalance <
    #     stats = [[index, stat, number] for index, stat, number in stats if any(lack.get(name, 0) > 0 for name in stat)]
    return methods


@trace
def prepare_data_set():
    statistic = Statistic()
    methods = dumpers.json_load(JODA_TIME_DATA_SET)
    print("raw methods", len(methods))
    methods = (filters.apply(method) for method in methods)
    methods = (join_java_doc(method) for method in methods if not empty(method))
    methods = (index_java_doc(method) for method in methods)
    methods = (parse_contract(method) for method in methods)
    methods = (method for method in methods if len(method["contract"]) > 0)
    methods = (statistic.before.consider(method) for method in methods)
    methods = (filter_contract(method) for method in methods)
    methods = (method for method in methods if len(method["contract"]) > 0)
    methods = (standardify_contract(method) for method in methods)
    methods = (method for method in methods if len(method["contract"]) > 0)
    methods = align_data_set(methods)
    methods = (statistic.after.consider(method) for method in methods)
    methods = (filter_contract_text(method) for method in methods)
    methods = (index_contract(method) for method in methods)
    batches = [build_batch(raw_batch) for raw_batch in batching(methods)]
    random.shuffle(batches)
    dumpers.pkl_dump(batches, ANALYSER_METHODS)
    print("batches", len(batches))
    statistic.show()


if __name__ == '__main__':
    init()
    prepare_data_set()
