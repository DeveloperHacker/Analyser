import itertools
from typing import Iterable, Any, List

import numpy as np
import tensorflow as tf

from constants import embeddings
from constants.tags import NOP, PARTS


def greedy_correct(targets, outputs):
    labels_targets, tokens_targets, strings_targets, strings_mask = targets
    labels, tokens, strings = outputs
    tokens_targets_values = np.asarray(embeddings.tokens().idx2emb)[tokens_targets]
    result_labels = []
    result_tokens = []
    result_strings = []
    result_strings_mask = []
    batch_size = tokens.shape[0]
    contract_size = tokens.shape[1]
    for i in range(batch_size):
        from_indexes = list(range(contract_size))
        to_indexes = list(range(contract_size))
        result_label = [None] * contract_size
        result_token = [None] * contract_size
        result_string = [None] * contract_size
        result_string_mask = [None] * contract_size
        while len(from_indexes) > 0:
            index_best_from_index = None
            index_best_to_index = None
            best_distance = None
            for j, from_index in enumerate(from_indexes):
                for k, to_index in enumerate(to_indexes):
                    from_output = tokens_targets_values[i][from_index]
                    to_output = tokens[i][to_index]
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
            result_label[best_to_index] = labels_targets[i][best_from_index]
            result_token[best_to_index] = tokens_targets[i][best_from_index]
            result_string[best_to_index] = strings_targets[i][best_from_index]
            result_string_mask[best_to_index] = strings_mask[i][best_from_index]
        result_labels.append(result_label)
        result_tokens.append(result_token)
        result_strings.append(result_string)
        result_strings_mask.append(result_string_mask)
    result_labels = np.asarray(result_labels)
    result_tokens = np.asarray(result_tokens)
    result_strings = np.asarray(result_strings)
    result_strings_mask = np.asarray(result_strings_mask)
    return result_labels, result_tokens, result_strings, result_strings_mask


def nearest_correct(targets, outputs, fine_weigh):
    labels_targets, tokens_targets, strings_targets, strings_mask = targets
    labels, tokens, strings = outputs
    tokens_targets_values = np.asarray(embeddings.tokens().idx2emb)[tokens_targets]
    result_labels = []
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
        result_labels.append(labels_targets[i][best_indices])
        result_tokens.append(tokens_targets[i][best_indices])
        result_strings.append(strings_targets[i][best_indices])
        result_strings_mask.append(strings_mask[i][best_indices])
    result_labels = np.asarray(result_labels)
    result_tokens = np.asarray(result_tokens)
    result_strings = np.asarray(result_strings)
    result_strings_mask = np.asarray(result_strings_mask)
    return result_labels, result_tokens, result_strings, result_strings_mask


def calc_accuracy(outputs_targets, outputs, labels_targets=None, labels=None):
    assert labels is None if labels_targets is None else labels is not None
    true_positive, true_negative, false_negative, false_positive = [], [], [], []
    batch_size = outputs.shape[0]
    contract_size = outputs.shape[1]
    nop = embeddings.tokens().get_index(NOP)
    is_nop = lambda condition: all(token == nop for token in condition)
    for i in range(batch_size):
        output_nops = [j for j in range(contract_size) if is_nop(outputs[i][j])]
        target_nops = [j for j in range(contract_size) if is_nop(outputs_targets[i][j])]
        output_indexes = [j for j in range(contract_size) if j not in output_nops]
        target_indexes = [j for j in range(contract_size) if j not in target_nops]
        _true_accept = 0
        _false_accept = 0
        for j in output_indexes:
            output_condition = outputs[i][j]
            if labels is not None:
                label = labels[i][j]
            index = None
            for k in range(len(target_indexes)):
                output_target_condition = outputs_targets[i][target_indexes[k]]
                if labels_targets is not None:
                    label_target = labels_targets[i][j]
                if all(token in output_target_condition for token in output_condition):
                    cond = labels_targets is not None
                    if cond and label == label_target or not cond:
                        index = k
                        break
            if index is None:
                _false_accept += 1
            else:
                del target_indexes[index]
                _true_accept += 1
        true_positive.append(_true_accept)
        true_negative.append(min(len(output_nops), len(target_nops)))
        false_negative.append(len(target_indexes))
        false_positive.append(_false_accept)
    return true_positive, true_negative, false_negative, false_positive


def transpose_mask(attention_mask, num_heads):
    """
        `[a x b x c]` is tensor with shape a x b x c

        batch_size, root_time_steps, num_decoders, num_heads, num_attentions is scalars
        attn_length is array of scalars

        Input:
        attention[i] is `[root_time_steps x bach_size x attn_length[i]]`
        attentions is [attention[i] for i in range(num_attentions) for j in range(num_heads)]
        attention_mask is [attentions for i in range(num_decoders)]

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
    def merge1(pack):
        return pack[0]

    # Merge stacked-decoding artifacts
    def merge2(pack):
        return pack[0]

    merged_attention_mask = []
    for attentions in attention_mask:
        merged_attentions = []
        for attention in chunks(attentions, num_heads):
            attention = merge1(np.asarray(attention).transpose([0, 2, 1, 3]))
            merged_attentions.append(attention.tolist())
        merged_attention_mask.append(merged_attentions)
    merged_attention_mask = np.asarray(merged_attention_mask)
    shape_length = len(merged_attention_mask.shape)
    merged_attention_mask = merged_attention_mask.transpose([0, 2, 3, 1, *range(4, shape_length)])
    merged_attention_mask = merge2(merged_attention_mask)
    return merged_attention_mask


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

    def cut(string: str, text_size: int):
        def length(word: str):
            pattern = re.compile("(\33\[\d+m)")
            found = re.findall(pattern, word)
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
    for i, label in enumerate(PARTS):
        maximum = np.max(words_weighs[i])
        minimum = np.min(words_weighs[i])
        mean = np.mean(words_weighs[i])
        variance = np.var(words_weighs[i])
        text = "m[W] = {:.4f}, d[W] = {:.4f}, min(W) = {:.4f}, max(W) = {:.4f}"
        text = text.format(mean, variance, minimum, maximum)
        formatter.print("", "")
        formatter.print("", " " + next(cut(text, text_size)))
        words = (embeddings.words().get_name(i) for i in indexed_doc[i])
        weighs = top_k_normalization(6, words_weighs[i])
        # weighs = normalization(words_weighs[i])
        words = colorize_words(words, weighs)
        for line in cut(" ".join(words), text_size):
            formatter.print(label, " " + line)


def cross_entropy_loss(targets, logits):
    with tf.variable_scope("cross_entropy_loss"):
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=targets, logits=logits)
        loss = tf.reduce_mean(loss, list(range(1, len(loss.shape))))
    return loss


def l2_loss(variables):
    with tf.variable_scope("l2_loss"):
        loss = tf.reduce_sum([tf.nn.l2_loss(variable) for variable in variables])
    return loss
