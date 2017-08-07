import random
from typing import Iterable, List, Any

import numpy as np
from contracts.guides.AstBfsGuide import AstBfsGuide
from contracts.guides.AstDfsGuide import AstDfsGuide
from contracts.nodes import StringNode
from contracts.nodes.Ast import Ast
from contracts.nodes.Node import Node
from contracts.parser import Parser
from contracts.tokens import Predicates, Markers
from contracts.tokens.Labels import WEAK
from contracts.visitors.AstCompiler import AstCompiler
from contracts.visitors.AstVisitor import AstVisitor
from pyparsing import ParseException

from configurations.constants import OUTPUT_TYPE, BATCH_SIZE
from configurations.logger import info_logger
from configurations.tags import PAD, NOP, NEXT, PARTS
from seq2seq import Embeddings
from utils import anonymizers


def empty(method):
    for label, text in method["java-doc"].items():
        if len(text) > 0:
            return False
    return True


def join_java_doc(method):
    java_doc = {label: " ".join(text) for label, text in method["java-doc"].items()}
    method["java-doc"] = java_doc
    return method


def apply_anonymizers(method):
    params = [param["name"] for param in method["description"]["parameters"]]
    java_doc = {label: anonymizers.apply(text, params) for label, text in method["java-doc"].items()}
    method["java-doc"] = java_doc
    return method


def one_line_doc(method):
    java_doc = (method["java-doc"][label].strip() for label in PARTS)
    java_doc = (" %s " % NEXT).join(text for text in java_doc if len(text) > 0)
    method["java-doc"] = java_doc
    return method


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
        if node.token == Predicates.NOT_EQUAL:
            assert node.children is not None
            assert len(node.children) == 2
            left = node.children[0]
            right = node.children[1]
            if left.token == Markers.FALSE:
                node = right
            elif right.token == Markers.FALSE:
                node = left
            elif left.token == Markers.TRUE:
                node.token = Predicates.EQUAL
                left.token = Markers.FALSE
                node.children = [right, left]
            elif right.token == Markers.TRUE:
                node.token = Predicates.EQUAL
                right.token = Markers.FALSE
        if node.token == Predicates.EQUAL:
            assert node.children is not None
            assert len(node.children) == 2
            left = node.children[0]
            right = node.children[1]
            if left.token == Markers.TRUE:
                node = right
            elif right.token == Markers.TRUE:
                node = left
            elif left.token == Markers.FALSE:
                node.children = [right, left]
        node.parent = parent
        return node

    def expand(node: Node):
        children = (Markers.PARAM, Markers.PARAM_0, Markers.PARAM_1,
                    Markers.PARAM_2, Markers.PARAM_3, Markers.PARAM_4,
                    Markers.RESULT, Markers.STRING, Predicates.GET)
        if node.token in children and (node.parent is None or node.parent.token == Predicates.FOLLOW):
            node = Node(Predicates.EQUAL, (node, Node(Markers.TRUE)), node.parent)
        return node

    reducer = AstDfsGuide(AstMapper(reduce))
    expander = AstDfsGuide(AstMapper(expand))
    compiler = AstDfsGuide(AstCompiler())
    forest = (Parser.parse_tree(*args) for args in method["contract"])
    forest = (reducer.accept(tree) for tree in forest)
    forest = list(expander.accept(tree) for tree in forest)
    method["contract"] = [compiler.accept(tree) for tree in forest]
    return method


def filter_contract(method):
    method["contract"] = [(label, tokens, strings) for label, tokens, strings in method["contract"] if label != WEAK]
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
            string = anonymizers.apply(string, self._params)
            node.words = string.split(" ")

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
    try:
        forest = Parser.parse("\n".join(method["contract"]))
    except ParseException as ex:
        for i, line in enumerate(method["contract"]):
            info_logger.info(line)
            if i + 1 == ex.lineno:
                info_logger.info("~" * ex.col + "^")
        raise ex
    except AssertionError as ex:
        for line in method["contract"]:
            info_logger.info(line)
        raise ex
    compiler = AstDfsGuide(AstCompiler())
    method["contract"] = [compiler.accept(tree) for tree in forest]
    return method


def index_contract(method):
    result = []
    for label, tokens, strings in method["contract"]:
        label = Embeddings.labels().get_index(label.name)
        tokens = [Embeddings.tokens().get_index(token.name) for token in tokens]
        strings = {idx: [Embeddings.words().get_index(word) for word in string] for idx, string in strings.items()}
        result.append((label, tokens, strings))
    method["contract"] = result
    return method


def index_java_doc(method):
    java_doc = (word.strip() for word in method["java-doc"].split(" "))
    java_doc = tuple(Embeddings.words().get_index(word) for word in java_doc if len(word) > 0)
    method["java-doc"] = java_doc
    return method


def build_batch(methods: List[dict]):
    pad = Embeddings.words().get_index(PAD)
    nop = Embeddings.tokens().get_index(NOP)

    inputs = [method["java-doc"] for method in methods]
    inputs_length = [len(line) for line in inputs]
    inputs_steps = max(inputs_length)
    inputs = [list(line) + [pad] * (inputs_steps + 1 - len(line)) for line in inputs]
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
    num_conditions = np.max(num_conditions)
    sequence_length = np.max(sequence_length)
    string_length = np.max(strings_lengths) + 1
    tree_depth = np.int32(np.ceil(np.log2(sequence_length)))
    if OUTPUT_TYPE == "tree":
        sequence_length = 2 ** tree_depth - 1

    strings = []
    tokens = []
    for method in methods:
        _strings = [[[-1 for _ in range(string_length)] for _ in range(sequence_length)] for _ in range(num_conditions)]
        _tokens = [[nop for _ in range(sequence_length)] for _ in range(num_conditions)]
        for i, (raw_label, raw_instructions, raw_strings) in enumerate(method["contract"]):
            _tokens[i][:len(raw_instructions)] = raw_instructions
            _tokens[i][len(raw_instructions):] = [nop] * (sequence_length - len(raw_instructions))
            for idx, raw_string in raw_strings.items():
                _strings[i][idx][:len(raw_string)] = raw_string
                _strings[i][idx][len(raw_string)] = pad
        strings.append(_strings)
        tokens.append(_tokens)
    tokens = np.asarray(tokens)
    strings = np.asarray(strings)

    inputs = (inputs, inputs_length)
    outputs = (tokens, strings)
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
