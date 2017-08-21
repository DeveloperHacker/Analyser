import enum
import itertools
import os
import re
from typing import Iterable, Any, List

import numpy as np
import tensorflow as tf
from contracts import Tokens, Types, Decompiler
from contracts.DfsGuide import DfsGuide
from contracts.Node import Node
from contracts.Token import Token
from contracts.Tree import Tree
from contracts.TreeVisitor import TreeVisitor

from analyser import Embeddings
from contants import NEXT, PAD, NOP
from logger import logger
from prepares import MAX_PARAM
from utils import Score
from utils.Formatter import Formatter
from utils.Style import Styles
from utils.wrappers import static


def cross_entropy_loss(targets, logits, default: int = None):
    with tf.variable_scope("cross_entropy_loss"):
        if default is not None:
            with tf.variable_scope("Masking"):
                output_size = logits.get_shape()[-1].value
                default_value = tf.one_hot(default, output_size) * output_size
                boolean_mask = tf.equal(targets, -1)
                W = tf.to_int32(boolean_mask)
                targets = (1 - W) * targets + W * default
                W = tf.to_float(tf.expand_dims(W, -1))
                logits = (1 - W) * logits + W * default_value
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=targets, logits=logits)
        loss = tf.reduce_sum(loss, list(range(1, len(loss.shape))))
    return loss


def l2_loss(variables):
    with tf.variable_scope("l2_loss"):
        loss = tf.reduce_sum([tf.nn.l2_loss(variable) for variable in variables])
    return loss


def greedy_correct(targets, outputs, dependencies):
    num_dependencies = len(dependencies)
    batch_size = outputs.shape[0]
    contract_size = outputs.shape[1]
    tokens_targets_values = np.asarray(Embeddings.tokens().idx2emb)[targets]
    result_tokens = []
    result_dependencies = [[] for _ in range(num_dependencies)]
    for i in range(batch_size):
        from_indexes = list(range(contract_size))
        to_indexes = list(range(contract_size))
        result_token = [None] * contract_size
        _result_dependencies = [[None] * contract_size for _ in range(num_dependencies)]
        while len(from_indexes) > 0:
            index_best_from_index = None
            index_best_to_index = None
            best_distance = None
            for j, from_index in enumerate(from_indexes):
                for k, to_index in enumerate(to_indexes):
                    from_output = tokens_targets_values[i][from_index]
                    to_output = outputs[i][to_index]
                    distance = np.linalg.norm(from_output - to_output)
                    if best_distance is None or distance < best_distance:
                        index_best_from_index = j
                        index_best_to_index = k
                        best_distance = distance
            best_from_index = from_indexes[index_best_from_index]
            best_to_index = to_indexes[index_best_to_index]
            del from_indexes[index_best_from_index]
            del to_indexes[index_best_to_index]
            result_token[best_to_index] = targets[i][best_from_index]
            for j in range(num_dependencies):
                _result_dependencies[j][best_to_index] = dependencies[j][i][best_from_index]
        result_tokens.append(result_token)
        for j in range(num_dependencies):
            result_dependencies[j].append(_result_dependencies[j])
    result_tokens = np.asarray(result_tokens)
    result_dependencies = [np.asarray(result_dependencies[i]) for i in range(num_dependencies)]
    return result_tokens, result_dependencies


def nearest_correct(targets, outputs, fine_weigh):
    tokens_targets, strings_targets, strings_mask = targets
    tokens, strings = outputs
    tokens_targets_values = np.asarray(Embeddings.tokens().idx2emb)[tokens_targets]
    result_tokens = []
    result_strings = []
    result_strings_mask = []
    batch_size = tokens.shape[0]
    contract_size = tokens.shape[1]
    for i in range(batch_size):
        default_indexes = np.arange(contract_size)
        best_indices = None
        for indexes in itertools.permutations(default_indexes):
            list_indexes = list(indexes)
            perm = tokens_targets_values[i][list_indexes]
            distance = np.linalg.norm(perm[1:] - tokens[i][1:])
            fine = fine_weigh * np.linalg.norm(default_indexes - indexes)
            distance = distance + fine
            if best_distance is None or distance < best_distance:
                best_indices = list_indexes
                best_distance = distance
        result_tokens.append(tokens_targets[i][best_indices])
        result_strings.append(strings_targets[i][best_indices])
        result_strings_mask.append(strings_mask[i][best_indices])
    result_tokens = np.asarray(result_tokens)
    result_strings = np.asarray(result_strings)
    result_strings_mask = np.asarray(result_strings_mask)
    return result_tokens, result_strings, result_strings_mask


def transpose_attention(attentions, num_heads=1):
    """
        `[a x b x c]` is tensor with shape a x b x c

        batch_size, root_time_steps, num_decoders, num_heads, num_attentions is scalars
        attn_length is array of scalars

        Input:
        attention[i] is `[root_time_steps x bach_size x attn_length[i]]`
        attention_mask is [attention[i] for i in range(num_attentions) for j in range(num_heads)]

        Output:
        attention is [`[attn_length[i]]` for i in range(num_attentions)]
        attentions is [attention for i in range(root_time_steps)]
        attention_mask is [attentions for i in range(batch_size)]
    """

    def chunks(iterable: Iterable[Any], block_size: int) -> Iterable[List[Any]]:
        result = []
        for element in iterable:
            result.append(element)
            if len(result) == block_size:
                yield result
                result = []
        if len(result) > 0:
            yield result

    # Merge multi-heading artifacts
    def merge(pack):
        return pack[0]

    attentions = [merge(attention) for attention in chunks(attentions, num_heads)]
    attentions = np.asarray(attentions)
    attentions = attentions.transpose([2, 1, 0, *range(3, len(attentions.shape))])
    return attentions


class Dropper(TreeVisitor):
    def __init__(self):
        super().__init__(DfsGuide())

    def visit_string(self, depth: int, node: Node, parent: Node):
        node.token = Token(Types.STRING, Types.STRING)


def calc_scores(tokens_targets, tokens, strings_targets, strings):
    def matrix_zip(depth, arrays: list) -> list:
        assert len(arrays) > 0
        if depth <= 0:
            return arrays
        length = len(arrays[0])
        get = lambda i: [array[i] for array in arrays]
        result = [matrix_zip(depth - 1, get(i)) for i in range(length)]
        return result

    def matrix_map(depth, array, mapper):
        if depth <= 0:
            return mapper(array)
        length = len(array)
        result = [matrix_map(depth - 1, array[i], mapper) for i in range(length)]
        return result

    def matrix_starmap(depth, array, mapper):
        _ = lambda x: mapper(*x)
        return matrix_map(depth, array, _)

    def matrix_parse(depth, _tokens, _strings):
        def _(tokens, strings) -> Tree:
            raw_tokens = []
            for token, string in zip(tokens, strings):
                if token == nop:
                    continue
                token = Embeddings.tokens().get_name(token)
                if token == Tokens.PARAM:
                    token = Tokens.PARAM + "[%d]" % MAX_PARAM
                if token == Types.STRING:
                    string = (word for word in string if word != pad)
                    token = " ".join(Embeddings.words().get_name(word) for word in string)
                    token = '"%s"' % token.replace('"', "'")
                raw_tokens.append(token)
            if tokens[0] == nop:
                raw_tokens = []
            else:
                raw_tokens.insert(0, Tokens.STRONG)
                raw_tokens.insert(0, Tokens.ROOT)
            result = list(Decompiler.typing(raw_tokens))
            try:
                tree = Decompiler.bfs(result)
            except Exception:
                token = Token(Tokens.ROOT, Types.ROOT)
                root = Node(token)
                tree = Tree(root)
            return tree

        zipped = matrix_zip(depth, (_tokens, _strings))
        mapped = matrix_starmap(depth, zipped, _)
        return mapped

    def matrix_drop(depth, array):
        def _(tree: Tree):
            tree = tree.clone()
            Dropper().accept(tree)
            return tree

        mapped = matrix_map(depth, array, _)
        mapped = matrix_flatten(depth, mapped)
        return mapped

    def matrix_flatten(depth, array):
        mapped = matrix_map(depth, array, str)
        return mapped

    nop = Embeddings.tokens().get_index(NOP)
    pad = Embeddings.words().get_index(PAD)
    depth = 2
    trees_targets = matrix_parse(depth, tokens_targets, strings_targets)
    trees = matrix_parse(depth, tokens, strings)
    code_targets = matrix_flatten(depth, trees_targets)
    code = matrix_flatten(depth, trees)
    contracts_targets = matrix_drop(depth, trees_targets)
    contracts = matrix_drop(depth, trees)
    tokens_scores = Score.calc(tokens_targets, tokens, None, nop)
    strings_scores = Score.calc(strings_targets, strings, -1, pad)
    contracts_scores = Score.calc(contracts_targets, contracts, None, Tokens.ROOT)
    code_scores = Score.calc(code_targets, code, None, Tokens.ROOT)
    return tokens_scores, strings_scores, contracts_scores, code_scores


def print_diff(inputs, attention, raw_tokens, tokens_targets, tokens, strings_targets, strings):
    class Align(enum.Enum):
        left = enum.auto()
        right = enum.auto()
        center = enum.auto()

    @static(pattern=re.compile("(\33\[\d+m)"))
    def cut(string: str, text_size: int, align: Align):
        def length(word: str):
            found = re.findall(cut.pattern, word)
            length = 0 if found is None else len("".join(found))
            return len(word) - length

        def chunks(line: str, max_line_length: int) -> Iterable[str]:
            words = line.split(" ")
            result = []
            result_length = 0
            for word in words:
                word_length = length(word)
                if result_length + word_length + 1 > max_line_length:
                    yield " ".join(result) + " " * (text_size - result_length)
                    result_length = 0
                    result = []
                result_length += word_length + 1
                result.append(word)
            yield " ".join(result) + " " * (text_size - result_length)

        lines = (" " + sub_line for line in string.split("\n") for sub_line in chunks(line, text_size - 1))
        return lines

    def print_doc(formatter, indexed_doc, words_weighs):
        STYLES = (Styles.foreground.gray,
                  Styles.bold % Styles.foreground.cyan,
                  Styles.bold % Styles.foreground.blue,
                  Styles.bold % Styles.foreground.magenta,
                  Styles.bold % Styles.foreground.red,
                  Styles.bold % Styles.foreground.yellow,
                  Styles.bold % Styles.foreground.green)

        def normalization(values: Iterable[float]) -> np.array:
            values = np.asarray(values)
            max_value = np.max(values)
            min_value = np.min(values)
            diff = (lambda diff: diff if diff > 0 else 1)(max_value - min_value)
            return (values - min_value) / diff

        def top_k_normalization(k: int, values: Iterable[float]) -> np.array:
            values = np.asarray(list(values))
            indices = np.argsort(values)[-k:]
            values = np.zeros(len(values))
            for i, j in enumerate(indices):
                values[j] = (i + 1) / k
            return values

        def split(words, *patterns):
            result = []
            for word in words:
                if any(re.findall(pattern, word) for pattern in patterns):
                    yield result
                    result = []
                else:
                    result.append(word)
            yield result

        for weighs in words_weighs:
            maximum = np.max(weighs)
            minimum = np.min(weighs)
            mean = np.mean(weighs)
            variance = np.var(weighs)
            text = "m[W] = {:.4f}, d[W] = {:.4f}, min(W) = {:.4f}, max(W) = {:.4f}"
            formatter.print(text.format(mean, variance, minimum, maximum))
            words = (Embeddings.words().get_name(index) for index in indexed_doc)
            # weighs = top_k_normalization(6, weighs)
            weighs = normalization(weighs)
            colorized = []
            for word, weigh in zip(words, weighs):
                style = STYLES[-1]
                for i, color in enumerate(STYLES):
                    if weigh < (i + 1) / len(STYLES):
                        style = color
                        break
                colorized.append(style % word)
            for text in split(colorized, NEXT, PAD):
                if len(text) == 0:
                    continue
                for line in cut(" ".join(text), formatter.size - 2, Align.left):
                    formatter.print(line)

    def print_raw_tokens(formatter, raw_tokens):
        matrix = [[None for _ in range(len(raw_tokens))] for _ in range(len(Embeddings.tokens()))]
        for j, raw_token in enumerate(raw_tokens):
            color0 = lambda x: Styles.background.light_yellow if x > 1e-2 else Styles.foreground.gray
            color1 = lambda x, is_max: Styles.background.light_red if is_max else color0(x)
            color = lambda x, is_max: color1(x, is_max) % "%.3f" % x
            for i, value in enumerate(raw_token):
                matrix[i][j] = color(value, i == np.argmax(raw_token))
        for i, token in enumerate(Embeddings.tokens().idx2name):
            text = " ".join(matrix[i])
            for line in cut(text, formatter.row_size(-1), Align.left):
                formatter.print(token, line)

    def print_strings(formatter, tokens, strings, strings_targets):
        for token, string, target in zip(tokens, strings, strings_targets):
            token = Embeddings.tokens().get_name(token)
            string = (Embeddings.words().get_name(index) if index >= 0 else " " for index in string)
            color = lambda skip: Styles.foreground.gray if skip else Styles.bold
            string = (color(index == -1) % word for word, index in zip(string, target))
            for line in cut(" ".join(string), formatter.row_size(-1), Align.left):
                formatter.print(token, line)

    formatter0 = Formatter(("tokens", "strings"), ("s", "s"), (30, 100))
    formatter1 = Formatter(["text"], ["s"], [formatter0.size - 2])
    batch_size = len(tokens)
    for i in range(batch_size):
        num_conditions = len(tokens[i])
        formatter0.print_upper_delimiter()
        for j in range(num_conditions):
            print_strings(formatter0, tokens[i][j], strings[i][j], strings_targets[i][j])
            formatter0.print_delimiter()
            print_strings(formatter0, tokens_targets[i][j], strings_targets[i][j], strings_targets[i][j])
            formatter0.print_delimiter()
            print_raw_tokens(formatter0, raw_tokens[i][j])
            formatter1.print_delimiter()
            print_doc(formatter1, inputs[i], attention[i][j])
            if j < num_conditions - 1:
                formatter1.print_delimiter()
        formatter1.print_lower_delimiter()


def print_scores(scores) -> (Score, Score, Score, Score):
    def to_short_tuple(score: Score):
        f1 = score.F_score(1) * 100
        acc = score.accuracy * 100
        jcc = score.jaccard * 100
        return f1, acc, jcc

    tokens_scores, strings_scores, templates_scores, codes_scores = zip(*scores)
    formatter = Formatter(("", "F1", "Accuracy", "Jaccard"), ("s", ".1f", ".1f", ".1f"), (20, 20, 20, 20))
    formatter.raw_print = logger.error
    formatter.print_head()
    tokens_score = Score.concat(tokens_scores)
    formatter.print("Tokens", *to_short_tuple(tokens_score))
    strings_score = Score.concat(strings_scores)
    formatter.print("Strings", *to_short_tuple(strings_score))
    templates_score = Score.concat(templates_scores)
    formatter.print("Templates", *to_short_tuple(templates_score))
    codes_score = Score.concat(codes_scores)
    formatter.print("Codes", *to_short_tuple(codes_score))
    formatter.print_lower_delimiter()
    return tokens_score, strings_score, templates_score, codes_score


def newest(path: str, filtrator):
    names = [path + "/" + name for name in os.listdir(path) if filtrator(path, name)]
    names.sort(key=os.path.getmtime, reverse=True)
    assert len(names) > 0, "Saves in dir '%s' hasn't found" % path
    return names[0]
