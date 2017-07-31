import itertools
import random
from multiprocessing import Value
from multiprocessing.pool import Pool
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
from contracts.visitors.AstVisitor import AstVisitor

from configurations.constants import BATCH_SIZE, OUTPUT_TYPE
from configurations.paths import ANALYSER_DATA_SET, ANALYSER_RAW_DATA_SET
from configurations.tags import PARTS, PAD, NOP
from generate_embeddings import join_java_doc
from seq2seq import Embeddings
from utils import anonymizers, dumpers
from utils.Formatter import Formatter
from utils.wrappers import Timer


class Counter:
    def set_value(self, value: int):
        self._value = value

    def get_value(self) -> int:
        return self._value

    def __init__(self, initial: int):
        self._value = initial

    def increment(self) -> int:
        self._value += 1
        return self._value


class SynchronizedCounter:
    def set_value(self, value: int):
        self._value.value = value

    def get_value(self) -> int:
        return self._value.value

    def __init__(self, initial: int):
        self._value = Value("i", initial)

    def increment(self) -> int:
        with self._value.get_lock():
            self._value.value += 1
            result = self._value.value
        return result


class Accountant:
    def __init__(self):
        self._methods_counter = Counter(0)
        self._token_counters = {token: Counter(0) for token in itertools.chain(tokens.predicates(), tokens.markers())}
        self._label_counters = {token: Counter(0) for token in tokens.labels()}

    def considers(self, methods: Iterable[dict]) -> Iterable[dict]:
        return (self.consider(method) for method in methods)

    def consider(self, method):
        self._methods_counter.increment()
        for label, instructions, strings in method["contract"]:
            for instruction in instructions:
                token = instruction.token
                self._token_counters[token.name].increment()
            self._label_counters[label.name].increment()
        return method

    def get_tokens(self) -> Iterable[Tuple[str, int]]:
        return ((name, counter.get_value()) for name, counter in self._token_counters.items())

    def get_labels(self) -> Iterable[Tuple[str, int]]:
        return ((name, counter.get_value()) for name, counter in self._label_counters.items())

    def get_num_tokens(self) -> int:
        return sum(counter.get_value() for counter in self._token_counters.values())

    def get_num_labels(self):
        return sum(counter.get_value() for counter in self._label_counters.values())

    def get_num_methods(self) -> int:
        return self._methods_counter.get_value()


class Statistic:
    def __init__(self):
        self.num_methods = 0
        self.num_batches = 0
        self.accountant_ids = []
        self._accountants = {}

    def accountant(self, accountant_id: str):
        accountant = Accountant()
        self.accountant_ids.append(accountant_id)
        self._accountants[accountant_id] = accountant
        return accountant

    def get_accountants(self) -> Iterable[Accountant]:
        for accountant_id in self.accountant_ids:
            yield self._accountants[accountant_id]

    def show(self):
        num_accountants = len(self.accountant_ids)
        assert num_accountants > 0
        heads = ("name", *self.accountant_ids)
        formats = ["s"] + ["d"] * num_accountants
        sizes = [15] * (num_accountants + 1)
        formatter = Formatter(heads, formats, sizes, range(num_accountants + 1))
        formatter.raw_print("Number of methods: %d" % self.num_methods)
        formatter.raw_print("Number of batches: %d" % self.num_batches)
        formatter.print_head()
        formatter.print("methods", *(step.get_num_methods() for step in self.get_accountants()))
        formatter.print("labels", *(step.get_num_labels() for step in self.get_accountants()))
        formatter.print("tokens", *(step.get_num_tokens() for step in self.get_accountants()))
        formatter.print_delimiter()
        values = []
        for labels in zip(*(step.get_labels() for step in self.get_accountants())):
            assert all(labels[0][0] == label[0] for label in labels)
            values.append((labels[0][0], *(label[1] for label in labels)))
        for name, *args in sorted(values, key=lambda x: x[1], reverse=True):
            formatter.print(name, *args)
        formatter.print_delimiter()
        values = []
        for tokens in zip(*(step.get_tokens() for step in self.get_accountants())):
            assert all(tokens[0][0] == token[0] for token in tokens)
            values.append((tokens[0][0], *(token[1] for token in tokens)))
        for name, *args in sorted(values, key=lambda x: x[1], reverse=True):
            formatter.print(name, *args)
        formatter.print_lower_delimiter()


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


def standardify_contract(method):
    reducer = AstDfsGuide(AstMapper(reduce))
    expander = AstDfsGuide(AstMapper(expand))
    compiler = AstDfsGuide(AstCompiler())
    forest = (Parser.parse_tree(*args) for args in method["contract"])
    forest = (reducer.accept(tree) for tree in forest)
    forest = list(expander.accept(tree) for tree in forest)
    method["contract"] = [compiler.accept(tree) for tree in forest]
    return method


def filter_contract(method):
    method["contract"] = [
        (label, instructions, strings)
        for label, instructions, strings in method["contract"]
        if label != WEAK
    ]
    return method


def chunks(iterable: Iterable[Any], block_size: int):
    result = []
    for element in iterable:
        result.append(element)
        if len(result) == block_size:
            yield result
            result = []
    if len(result) > 0:
        yield result


def batching(methods: Iterable[dict]):
    methods = list(methods)
    random.shuffle(methods)
    return (chunk for chunk in chunks(methods, BATCH_SIZE) if len(chunk) == BATCH_SIZE)


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
        string = anonymizers.apply_anonymizers(string, self._params)
        node.words = string.split(" ")


def filter_contract_text(method):
    filtrator = AstDfsGuide(StringFiltrator(method))
    if OUTPUT_TYPE in ("tree", "bfs_sequence"):
        compiler = AstBfsGuide(AstCompiler())
    elif OUTPUT_TYPE == "dfs_sequence":
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
        label = Embeddings.labels().get_index(label.name)
        instructions = [
            Embeddings.tokens().get_index(instruction.token.name)
            for instruction in instructions
        ]
        strings = {
            idx: [Embeddings.words().get_index(word) for word in string]
            for idx, string in strings.items()
        }
        result.append((label, instructions, strings))
    method["contract"] = result
    return method


def index_java_doc(method):
    result = {}
    for label, text in method["java-doc"].items():
        split = (word.strip() for word in text.split(" "))
        indexes = tuple(Embeddings.words().get_index(word) for word in split if len(word) > 0)
        result[label] = indexes
    method["java-doc"] = result
    return method


def build_batch(methods: List[dict]):
    pad = Embeddings.words().get_index(PAD)
    nop = Embeddings.tokens().get_index(NOP)

    inputs = [list(itertools.chain(*(method["java-doc"][label] for label in PARTS))) for method in methods]
    inputs_length = [len(line) for line in inputs]
    inputs_steps = max(inputs_length)
    inputs = [line + [pad] * (inputs_steps + 1 - len(line)) for line in inputs]
    inputs = np.asarray(inputs)
    inputs_length = np.asarray(inputs_length)

    num_conditions = []
    sequence_length = []
    strings_lengths = [1]
    for method in methods:
        contract = method["contract"]
        num_conditions.append(len(contract))
        for raw_label, raw_instructions, raw_strings in contract:
            sequence_length.append(len(raw_instructions))
            strings_lengths.extend(len(string) for idx, string in raw_strings.items())
    num_conditions = max(num_conditions)
    sequence_length = max(sequence_length)
    tree_depth = int(np.ceil(np.log2(sequence_length)))
    string_length = max(strings_lengths)
    if OUTPUT_TYPE == "tree":
        sequence_length = 2 ** tree_depth - 1
    num_conditions = np.int32(num_conditions)
    sequence_length = np.int32(sequence_length)
    tree_depth = np.int32(tree_depth)
    string_length = np.int32(string_length)

    W_strings = []
    strings = []
    tokens = []
    labels = []
    empty_sequence = [nop] * sequence_length
    empty_string = [pad] * string_length
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
        W_strings.append(string_mask)
        strings.append(empty_strings)
        tokens.append(empty_tokens)
        labels.append(empty_labels)
    labels = np.asarray(labels)
    tokens = np.asarray(tokens)
    strings = np.asarray(strings)
    W_strings = np.expand_dims(np.expand_dims(np.asarray(W_strings), -1), -1)
    pad_one_hot = np.zeros(len(Embeddings.words()))
    pad_one_hot[pad] = 1
    B_strings = [[[[pad_one_hot] * string_length] * sequence_length] * num_conditions] * BATCH_SIZE
    B_strings = np.asarray(B_strings) * (1 - W_strings)

    inputs = (inputs, inputs_length)
    outputs = (labels, tokens, strings, W_strings, B_strings)
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


def prepare_data_set():
    statistic = Statistic()
    methods = dumpers.json_load(ANALYSER_RAW_DATA_SET)
    statistic.num_methods = len(methods)
    with Pool() as pool:
        with Timer("step 1") as timer:
            methods = pool.map(join_java_doc, methods)
            methods = pool.map(anonymizers.apply, methods)
            methods = pool.map(index_java_doc, methods)
            methods = pool.map(parse_contract, methods)
            methods = statistic.accountant(timer.name).considers(methods)
        with Timer("step 2") as timer:
            methods = (method for method in methods if len(method["contract"]) > 0)
            methods = statistic.accountant(timer.name).considers(methods)
        with Timer("step 3"):
            methods = pool.map(filter_contract, methods)
            methods = (method for method in methods if len(method["contract"]) > 0)
        with Timer("step 4"):
            methods = pool.map(standardify_contract, methods)
            methods = (method for method in methods if len(method["contract"]) > 0)
        with Timer("step 5") as timer:
            methods = align_data_set(methods)
            methods = statistic.accountant(timer.name).considers(methods)
        with Timer("step 6"):
            methods = pool.map(filter_contract_text, methods)
            methods = pool.map(index_contract, methods)
            batches = pool.map(build_batch, batching(methods))
    statistic.num_batches = len(batches)
    statistic.show()
    with Timer("step 7"):
        random.shuffle(batches)
    with Timer("step 8"):
        dumpers.pkl_dump(batches, ANALYSER_DATA_SET)


if __name__ == '__main__':
    with Timer("PREPARE DATA-SET"):
        prepare_data_set()
