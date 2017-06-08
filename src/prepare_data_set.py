import logging
import sys
from multiprocessing.pool import Pool

import numpy as np
from contracts.nodes.StringNode import StringNode
from contracts.nodes.WordNode import WordNode
from contracts.parser.Parser import Parser
from contracts.visitors.AstCompiler import AstCompiler
from contracts.visitors.AstVisitor import AstVisitor
from typing import List, Any, Iterable

from constants.analyser import BATCH_SIZE
from constants.embeddings import WordEmbeddings, TokenEmbeddings, NOP, PAD
from constants.paths import ANALYSER_PREPARE_DATA_SET_LOG, ANALYSER_METHODS, RAW_METHODS
from constants.tags import PARTS
from generate_embeddings import join_java_doc, empty
from utils import Filter, Dumper
from utils.wrapper import trace


class StringFiltrator(AstVisitor):
    def __init__(self, method):
        super().__init__()
        self.method = method

    def _visit_string(self, node: StringNode):
        string = " ".join(word.instance for word in node.children)
        params = [param["name"] for param in self.method["description"]["parameters"]]
        string = Filter.applyFiltersForString(string, params)
        node.children = [WordNode(word) for word in string.split(" ")]


@trace
def prepare_data_set():
    methods = Dumper.json_load(RAW_METHODS)
    with Pool() as pool:
        methods = pool.map(apply, methods)
        methods = (method for method in methods if method is not None)
        methods = pool.map(build_batch, batching(methods))
    Dumper.pkl_dump(methods, ANALYSER_METHODS)


def apply(method):
    method = parse_contract(method)
    if len(method["contract"]) <= 1: return None
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
    result = [len(method["java-doc"][label]) for label in PARTS]
    result.append(len(method["contract"]))
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
    tree = Parser.tree(method["contract"])
    filtrator = StringFiltrator(method)
    filtrator.accept(tree)
    compiler = AstCompiler()
    compiler.accept(tree)
    method["contract"] = compiler.instructions
    return method


def parse_contract(method):
    method["contract"] = Parser.parse("\n".join(method["contract"]))
    return method


def index_contract(method):
    result = []
    for instruction in method["contract"]:
        word_index = WordEmbeddings.get_index(instruction.word)
        token_index = TokenEmbeddings.get_index(instruction.token.name)
        result.append((word_index, token_index))
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
    inputs_steps = {label: np.asarray([len(method["java-doc"][label]) for method in methods]) for label in PARTS}
    output_steps = np.asarray([len(method["contract"]) for method in methods])
    inputs_steps = {label: np.max(inputs_steps[label]) for label in PARTS}
    output_steps = np.max(output_steps)

    inputs = {label: [] for label in PARTS}
    inputs_sizes = {label: [] for label in PARTS}
    word_target = []
    token_target = []
    for method in methods:
        for label in PARTS:
            line = method["java-doc"][label]
            inputs_sizes[label].append(len(line))
            line = list(line) + [WordEmbeddings.get_index(PAD) for _ in range(inputs_steps[label] + 1 - len(line))]
            inputs[label].append(line)
        words = []
        tokens = []
        for word, token in method["contract"]:
            words.append(word)
            tokens.append(token)
        words += [WordEmbeddings.get_index(PAD) for _ in range(output_steps + 1 - len(words))]
        tokens += [TokenEmbeddings.get_index(NOP) for _ in range(output_steps + 1 - len(tokens))]
        word_target.append(words)
        token_target.append(tokens)
    for label in PARTS:
        inputs[label] = np.transpose(np.asarray(inputs[label]), (1, 0))
    word_target = np.transpose(np.asarray(word_target), (1, 0))
    token_target = np.transpose(np.asarray(token_target), (1, 0))
    return inputs, inputs_sizes, word_target, token_target


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, filename=ANALYSER_PREPARE_DATA_SET_LOG)
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    prepare_data_set()
