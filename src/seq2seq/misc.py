import itertools
from typing import Iterable, Any, List

import numpy as np
import tensorflow as tf

from configurations.tags import NOP
from seq2seq import Embeddings
from utils.wrappers import static


def cross_entropy_loss(targets, logits):
    with tf.variable_scope("cross_entropy_loss"):
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=targets, logits=logits)
        loss = tf.reduce_mean(loss, list(range(1, len(loss.shape))))
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
                    # distance = np.sum(from_output * to_output)
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


def calc_accuracy(targets, outputs):
    length = outputs.shape[0]
    nop = Embeddings.tokens().get_index(NOP)
    is_nop = lambda condition: all(token == nop for token in condition)
    output_nops = [j for j in range(length) if is_nop(outputs[j])]
    target_nops = [j for j in range(length) if is_nop(targets[j])]
    output_indexes = [j for j in range(length) if j not in output_nops]
    target_indexes = [j for j in range(length) if j not in target_nops]
    true_negative = min(len(output_nops), len(target_nops))
    true_positive = 0
    false_positive = 0
    for j in output_indexes:
        output_condition = outputs[j]
        index = None
        for k in range(len(target_indexes)):
            output_target = targets[target_indexes[k]]
            if all(token in output_target for token in output_condition):
                index = k
                break
        if index is None:
            false_positive += 1
        else:
            del target_indexes[index]
            true_positive += 1
    false_negative = len(target_indexes)
    error = true_positive + false_negative + false_positive
    accuracy = true_positive / error
    error += true_negative
    true_positive /= error
    true_negative /= error
    false_negative /= error
    false_positive /= error
    return accuracy, (true_positive, true_negative, false_negative, false_positive)


def batch_accuracy(targets, outputs):
    results = (calc_accuracy(target, output) for target, output in zip(targets, outputs))
    results = zip(*((result[0], *result[1]) for result in results))
    accuracy, true_positive, true_negative, false_negative, false_positive = results
    return accuracy, (true_positive, true_negative, false_negative, false_positive)


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


def print_doc(formatter, indexed_doc, words_weighs):
    import re
    STYLES = (tuple(), (1, 106), (1, 97, 104), (1, 105), (1, 97, 101), (1, 103), (1, 102))

    def legend() -> str:
        size = (formatter.size - 4 - 14) // len(STYLES)
        arrow_begin = " " + "─" * (size - 1)
        arrow_body = "─" * size
        arrow_end = "─" * (size - 2) + "> "
        arrow = []
        for i, styles in enumerate(STYLES):
            text = arrow_begin if i == 0 else arrow_body if i < len(STYLES) - 1 else arrow_end
            stylized = "".join("\33[%dm" % style for style in styles) + text + "\33[0m"
            arrow.append(stylized)
        return "".join(arrow)

    def colorize_words(words: Iterable[str], weighs: Iterable[float]) -> str:
        result = []
        for word, weigh in zip(words, weighs):
            styles = STYLES[-1]
            for i, color in enumerate(STYLES):
                if weigh < (i + 1) / len(STYLES):
                    styles = color
                    break
            word = "".join("\33[%dm" % style for style in styles) + word + "\33[0m"
            result.append(word)
        return result

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

    @static(pattern=re.compile("(\33\[\d+m)"))
    def cut(string: str, text_size: int):
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
                if result_length + word_length > max_line_length:
                    yield " ".join(result) + " " * (text_size - result_length + 1)
                    result_length = 0
                    result = []
                result_length += word_length + 1
                result.append(word)
            yield " ".join(result) + " " * (text_size - result_length + 1)

        lines = (sub_line for line in string.split("\n") for sub_line in chunks(line, text_size))
        return lines

    text_size = formatter.row_size(-1) - 1
    formatter.print("", " " + next(cut(legend(), text_size)))
    for word_weight in words_weighs:
        maximum = np.max(word_weight)
        minimum = np.min(word_weight)
        mean = np.mean(word_weight)
        variance = np.var(word_weight)
        text = "m[W] = {:.4f}, d[W] = {:.4f}, min(W) = {:.4f}, max(W) = {:.4f}"
        text = text.format(mean, variance, minimum, maximum)
        formatter.print("", "")
        formatter.print("", " " + next(cut(text, text_size)))
        words = (Embeddings.words().get_name(index) for index in indexed_doc)
        weighs = top_k_normalization(6, word_weight)
        # weighs = normalization(words_weighs[i])
        words = colorize_words(words, weighs)
        for line in cut(" ".join(words), text_size):
            formatter.print("", " " + line)
