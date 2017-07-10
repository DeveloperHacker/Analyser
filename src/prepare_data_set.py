import os
import random
from multiprocessing import Value
from multiprocessing.pool import Pool
from typing import List, Any, Iterable

import numpy as np
from contracts.guides.AstBfsGuide import AstBfsGuide
from contracts.guides.AstDfsGuide import AstDfsGuide
from contracts.nodes.Ast import Ast
from contracts.nodes.StringNode import StringNode
from contracts.parser import Parser
from contracts.tokens import tokens
from contracts.tokens.Token import Token
from contracts.visitors.AstCompiler import AstCompiler
from contracts.visitors.AstEqualReducer import AstEqualReducer
from contracts.visitors.AstVisitor import AstVisitor

from config import init
from constants.analyser import BATCH_SIZE, SEED
from constants.embeddings import WordEmbeddings, TokenEmbeddings, PAD, NOP
from constants.paths import ANALYSER_METHODS, JODA_TIME_METHODS
from constants.tags import PARTS
from generate_embeddings import join_java_doc, empty
from utils import Filter, Dumper
from utils.Formatter import Formatter
from utils.wrapper import trace


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
        string = Filter.applyFiltersForString(string, self._params)
        node.words = string.split(" ")


class Statistic:
    def __init__(self):
        self.tokens_counters = {token: Value("i", 0) for token in tokens.instances()}
        self.num_raw_methods = None
        self.num_methods = None
        self.num_batches = None

    def count(self, token: Token):
        counter = self.tokens_counters[token.name]
        with counter.get_lock():
            counter.value += 1

    def num_tokens(self) -> int:
        return sum(counter.value for counter in self.tokens_counters.values())

    def show(self):
        method_concentration = self.num_methods / self.num_raw_methods * 100
        num_tokens = self.num_tokens()
        values = []
        for token_name, counter in self.tokens_counters.items():
            number = counter.value
            concentration = number / num_tokens * 100
            values.append((token_name, number, concentration))
        values = sorted(values, key=lambda x: x[1], reverse=True)
        formatter = Formatter(("Name", "Number", "Concentration"), ("s", "d", "s"), (20, 20, 20), (0, 1, 2))
        formatter.print_head()
        formatter.print("raw methods", self.num_raw_methods, "")
        formatter.print("methods", self.num_methods, "%.1f%%" % method_concentration)
        formatter.print("batches", self.num_batches, "")
        formatter.print("tokens", num_tokens, "")
        formatter.print_delimiter()
        for token_name, number, concentration in values:
            formatter.print(token_name, number, "%.1f%%" % concentration)
        formatter.print_lower_delimiter()


statistic = Statistic()


@trace
def prepare_data_set():
    methods = Dumper.json_load(JODA_TIME_METHODS)
    statistic.num_raw_methods = len(methods)
    with Pool() as pool:
        methods = pool.map(apply, methods)
        methods = [method for method in methods if method is not None]
        statistic.num_methods = len(methods)
        methods = pool.map(build_batch, batching(methods))
        statistic.num_batches = len(methods)
    random.shuffle(methods, lambda: random.Random(SEED).uniform(0, 1))
    Dumper.pkl_dump(methods, ANALYSER_METHODS)
    statistic.show()


def apply(method):
    try:
        method = parse_contract(method)
        if len(method["contract"]) == 0: return None
        method = filter_contract(method)
        if len(method["contract"]) == 0: return None
        method = standardify_contract(method)
        method = filter_contract_text(method)
        method = index_contract(method)
        method = Filter.apply(method)
        if empty(method): return None
        method = join_java_doc(method)
        method = index_java_doc(method)
    except Exception:
        raise ValueError()
    return method


def standardify_contract(method):
    reducer = AstDfsGuide(AstEqualReducer())
    compiler = AstDfsGuide(AstCompiler())
    forest = (Parser.parse_tree(*args) for args in method["contract"])
    forest = (reducer.accept(tree) for tree in forest)
    method["contract"] = [compiler.accept(tree) for tree in forest]
    return method


def filter_contract(method):
    new_tokens = {tokens.POST_THIS.name, tokens.PRE_THIS.name, tokens.THIS.name, tokens.GET.name}
    contract = []
    for label, instructions, strings in method["contract"]:
        instructions_names = set(instruction.token.name for instruction in instructions)
        intersection = instructions_names & new_tokens
        # if tokens.NULL.name in instructions_names:
        if len(intersection) == 0:
            contract.append((label, instructions, strings))
    method["contract"] = contract
    return method


def vectorize(method) -> List[int]:
    result = []
    # result.extend(len(method["java-doc"][label]) for label in PARTS)
    # result.append(len(method["contract"]))
    length = []
    outputs_steps = len(method["contract"])
    for label, instructions, strings in method["contract"]:
        length.append(len(instructions))
    length = max(length)
    depth = int(np.ceil(np.log2(length)))
    output_type = os.environ['OUTPUT_TYPE']
    if output_type == "tree":
        length = 2 ** depth
    elif output_type in ("bfs_sequence", "dfs_sequence"):
        length += 1
    result.append(outputs_steps)
    result.append(length)
    return result


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
    methods = ((vectorize(method), method) for method in methods)
    methods = sorted(methods, key=lambda x: np.linalg.norm(x[0]))
    methods = (method for vector, method in methods)
    return (chunk for chunk in chunks(methods, BATCH_SIZE) if len(chunk) == BATCH_SIZE)


def filter_contract_text(method):
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
    method["contract"] = Parser.parse("\n".join(method["contract"]))
    for label, instructions, strings in method["contract"]:
        for instruction in instructions:
            token = instruction.token
            statistic.count(token)
        statistic.count(label)
    return method


def index_contract(method):
    result = []
    for label, instructions, strings in method["contract"]:
        label = TokenEmbeddings.get_index(label.name)
        instructions = [
            TokenEmbeddings.get_index(instruction.token.name)
            for instruction in instructions
        ]
        strings = {
            idx: [WordEmbeddings.get_index(word) for word in string]
            for idx, string in strings.items()
        }
        result.append((label, instructions, strings))
    method["contract"] = result
    return method


def index_java_doc(method):
    result = {}
    for label, text in method["java-doc"].items():
        split = text.split(" ")
        indexes = tuple(WordEmbeddings.get_index(word) for word in split)
        result[label] = indexes
    method["java-doc"] = result
    return method


def build_batch(methods: List[dict]):
    inputs_steps = {
        label: max([len(method["java-doc"][label]) for method in methods])
        for label in PARTS
    }
    inputs = {label: [] for label in PARTS}
    inputs_sizes = {label: [] for label in PARTS}
    pad_index = WordEmbeddings.get_index(PAD)
    for method in methods:
        for label in PARTS:
            line = list(method["java-doc"][label])
            inputs_sizes[label].append(len(line))
            expected = inputs_steps[label] + 1 - len(line)
            line = line + [pad_index] * expected
            inputs[label].append(line)
    for label in PARTS:
        inputs[label] = np.transpose(np.asarray(inputs[label]), (1, 0))
    root_time_steps = []
    output_time_steps = []
    for method in methods:
        contract = method["contract"]
        root_time_steps.append(len(contract))
        for label, instructions, strings in contract:
            output_time_steps.append(len(instructions))
    root_time_steps = max(root_time_steps)
    output_time_steps = max(output_time_steps)
    depth = int(np.ceil(np.log2(output_time_steps)))
    output_type = os.environ['OUTPUT_TYPE']
    if output_type == "tree":
        output_time_steps = 2 ** depth - 1
    elif output_type in ("bfs_sequence", "dfs_sequence"):
        output_time_steps += 1
    outputs = []
    nop_index = TokenEmbeddings.get_index(NOP)
    empty_sequence = [nop_index for _ in range(output_time_steps + 1)]
    for method in methods:
        outputs.append([])
        for label, instructions, strings in method["contract"]:
            expected = output_time_steps - len(instructions)
            line = [label] + instructions + [nop_index] * expected
            outputs[-1].append(line)
        expected = root_time_steps - len(outputs[-1])
        outputs[-1].extend([empty_sequence] * expected)
    outputs = np.asarray(outputs)
    return inputs, inputs_sizes, outputs, root_time_steps, output_time_steps, depth


if __name__ == '__main__':
    init()
    prepare_data_set()
