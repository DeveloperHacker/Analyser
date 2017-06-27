import os
from multiprocessing.pool import Pool
from typing import List, Any, Iterable

import numpy as np
from contracts.guides.AstBfsGuide import AstBfsGuide
from contracts.guides.AstDfsGuide import AstDfsGuide
from contracts.nodes.Ast import Ast
from contracts.nodes.StringNode import StringNode
from contracts.parser.Parser import Parser
from contracts.visitors.AstCompiler import AstCompiler
from contracts.visitors.AstVisitor import AstVisitor

from config import init
from constants.analyser import BATCH_SIZE
from constants.embeddings import WordEmbeddings, TokenEmbeddings, PAD, NOP
from constants.paths import ANALYSER_METHODS, RAW_METHODS
from constants.tags import PARTS
from generate_embeddings import join_java_doc, empty
from utils import Filter, Dumper
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


@trace
def prepare_data_set():
    methods = Dumper.json_load(RAW_METHODS)
    with Pool() as pool:
        methods = pool.map(apply, methods)
        methods = [method for method in methods if method is not None]
        methods = pool.map(build_batch, batching(methods))
    Dumper.pkl_dump(methods, ANALYSER_METHODS)


def apply(method):
    method = parse_contract(method)
    if len(method["contract"]) == 0: return None
    method = simplify_contract(method)
    method = filter_contract_text(method)
    method = index_contract(method)
    method = Filter.apply(method)
    if empty(method): return None
    method = join_java_doc(method)
    method = index_java_doc(method)
    return method


def simplify_contract(method):
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
    dfs_guide = AstDfsGuide(StringFiltrator(method))
    output_type = os.environ['OUTPUT_TYPE']
    if output_type in ("tree", "bfs_sequence"):
        compile_guide = AstBfsGuide(AstCompiler())
    elif output_type == "dfs_sequence":
        compile_guide = AstDfsGuide(AstCompiler())
    forest = (Parser.parse_tree(*args) for args in method["contract"])
    forest = (dfs_guide.accept(tree) for tree in forest)
    method["contract"] = [compile_guide.accept(tree) for tree in forest]
    return method


def parse_contract(method):
    method["contract"] = Parser.parse("\n".join(method["contract"]))
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
