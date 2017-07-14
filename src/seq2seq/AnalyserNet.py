import itertools
import logging
import os
import random
import time

import numpy as np
import tensorflow as tf
from live_plotter.proxy.ProxyFigure import ProxyFigure
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.ops.rnn_cell_impl import GRUCell

from config import init
from constants.analyser import *
from constants.paths import ANALYSER, ANALYSER_METHODS, ANALYSER_SUMMARIES
from constants.tags import PARTS, NEXT, PAD, NOP
from seq2seq.Net import Net
from seq2seq.analyser_rnn import sequence_input, input_projection, sequence_output, tree_output
from utils import Dumper
from utils.Formatter import Formatter
from utils.SmoothedValue import SmoothedValue
from utils.wrapper import trace


def nearest_correct(labels_targets, outputs_targets, outputs, fine_weigh):
    outputs_targets_values = np.asarray(embeddings.tokens().idx2emb)[outputs_targets]
    result_labels = []
    result_outputs = []
    batch_size = outputs.shape[0]
    contract_size = outputs.shape[1]
    for i in range(batch_size):
        output = outputs[i]
        label_target = labels_targets[i]
        output_target = outputs_targets[i]
        output_target_values = outputs_targets_values[i]
        best_output = None
        best_distance = None
        default_indexes = np.arange(contract_size)
        for indexes in itertools.permutations(default_indexes):
            list_indexes = list(indexes)
            perm = output_target_values[list_indexes]
            distance = np.linalg.norm(perm[1:] - output[1:])
            fine = fine_weigh * np.linalg.norm(default_indexes - indexes)
            distance = distance + fine
            if best_distance is None or distance < best_distance:
                best_label = label_target[list_indexes]
                best_output = output_target[list_indexes]
                best_distance = distance
        result_labels.append(best_label)
        result_outputs.append(best_output)
    return np.asarray(result_labels), np.asarray(result_outputs)


def greedy_correct(labels_targets, outputs_targets, outputs):
    outputs_targets_values = np.asarray(embeddings.tokens().idx2emb)[outputs_targets]
    result_labels = []
    result_outputs = []
    batch_size = outputs.shape[0]
    contract_size = outputs.shape[1]
    for i in range(batch_size):
        output = outputs[i]
        label_target = labels_targets[i]
        output_target = outputs_targets[i]
        output_target_values = outputs_targets_values[i]
        from_indexes = list(range(contract_size))
        to_indexes = list(range(contract_size))
        result_label = [None] * contract_size
        result_output = [None] * contract_size
        while len(from_indexes) > 0:
            index_best_from_index = None
            index_best_to_index = None
            best_distance = None
            for j, from_index in enumerate(from_indexes):
                for k, to_index in enumerate(to_indexes):
                    from_output = output_target_values[from_index]
                    to_output = output[to_index]
                    distance = np.linalg.norm(from_output[1:] - to_output[1:])
                    if best_distance is None or distance < best_distance:
                        index_best_from_index = j
                        index_best_to_index = k
                        best_distance = distance
            best_from_index = from_indexes[index_best_from_index]
            best_to_index = to_indexes[index_best_to_index]
            del from_indexes[index_best_from_index]
            del to_indexes[index_best_to_index]
            result_label[best_to_index] = label_target[best_from_index]
            result_output[best_to_index] = output_target[best_from_index]
        result_labels.append(result_label)
        result_outputs.append(result_output)
    return np.asarray(result_labels), np.asarray(result_outputs)


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


def cross_entropy_loss(targets, logits):
    with vs.variable_scope("cross_entropy_loss"):
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=targets, logits=logits)
        loss = tf.reduce_mean(loss, list(range(1, len(loss.shape))))
    return loss


def l2_loss(variables):
    with vs.variable_scope("l2_loss"):
        loss = tf.reduce_sum([tf.nn.l2_loss(variable) for variable in variables])
    return loss


class AnalyserNet(Net):
    @trace
    def __init__(self):
        super().__init__("analyser", ANALYSER)
        with vs.variable_scope(self.name):
            _embeddings = tf.constant(np.asarray(embeddings.words().idx2emb))
            self.inputs = {}
            self.embeddings = {}
            self.inputs_sizes = {}
            cells_fw = {}
            cells_bw = {}
            for label in PARTS:
                label = label[1:]
                indexes = tf.placeholder(tf.int32, [BATCH_SIZE, None], "indexes_%s" % label)
                self.embeddings[label] = tf.gather(_embeddings, indexes)
                self.inputs[label] = indexes
                self.inputs_sizes[label] = tf.placeholder(tf.int32, [BATCH_SIZE], "input_sizes_%s" % label)
                cells_fw[label] = [GRUCell(ENCODER_STATE_SIZE) for _ in range(NUM_ENCODERS)]
                cells_bw[label] = [GRUCell(ENCODER_STATE_SIZE) for _ in range(NUM_ENCODERS)]
            root_cells = [GRUCell(ROOT_STATE_SIZE) for _ in range(NUM_DECODERS)]
            sequence_cells = [GRUCell(SEQUENCE_STATE_SIZE) for _ in range(NUM_DECODERS)]
            self.root_time_steps = tf.placeholder(tf.int32, [], "root_time_steps")
            self.output_time_steps = tf.placeholder(tf.int32, [], "output_time_steps")
            self.depth = tf.placeholder(tf.int32, [], "depth")
            projection = input_projection(self.embeddings, INPUT_SIZE, tf.float32)
            attention_states = sequence_input(cells_bw, cells_fw, projection, self.inputs_sizes, tf.float32)
            if os.environ['OUTPUT_TYPE'] == "tree":
                self.labels_logits, self.labels, self.outputs_logits, self.outputs = tree_output(
                    attention_states, root_cells, self.root_time_steps, ROOT_NUM_HEADS,
                    self.depth,
                    NUM_LABELS, NUM_TOKENS, tf.float32)
            elif os.environ['OUTPUT_TYPE'] in ("bfs_sequence", "dfs_sequence"):
                self.labels_logits, self.labels, self.outputs_logits, self.outputs = sequence_output(
                    attention_states, root_cells, self.root_time_steps, ROOT_NUM_HEADS,
                    sequence_cells, self.output_time_steps, SEQUENCE_NUM_HEADS,
                    NUM_LABELS, NUM_TOKENS, tf.float32)
            self.top_labels = tf.nn.top_k(self.labels, min(2, TOP))
            self.top_outputs = tf.nn.top_k(self.outputs, TOP)
            self.labels_targets = tf.placeholder(tf.int32, [BATCH_SIZE, None], "labels_targets")
            self.outputs_targets = tf.placeholder(tf.int32, [BATCH_SIZE, None, None], "outputs_targets")
            self.scope = vs.get_variable_scope().name
            self.accuracy_loss, self.loss = self.build_loss()
        self.optimizer = tf.train.AdamOptimizer().minimize(self.loss)
        self._data_set = Dumper.pkl_load(ANALYSER_METHODS)
        self.add_variable_summaries()
        self.summaries = tf.summary.merge_all()

    def add_variable_summaries(self):
        tf.summary.histogram("sum", tf.concat([tf.reshape(variable, [-1]) for variable in self.variables], 0))
        for variable in self.variables:
            tf.summary.histogram(variable.name, variable)

    def build_loss(self):
        with vs.variable_scope("loss"):
            labels_loss = cross_entropy_loss(self.labels_targets, self.labels_logits)
            outputs_loss = cross_entropy_loss(self.outputs_targets, self.outputs_logits)
            _l2_loss = L2_LOSS_WEIGHT * l2_loss(self.variables)
            accuracy_loss = labels_loss + outputs_loss
            loss = accuracy_loss + _l2_loss
        return accuracy_loss, loss

    def build_feed_dict(self, batch) -> dict:
        feed_dict = {}
        inputs, inputs_sizes, labels, outputs, root_time_steps, output_time_steps, depth = batch
        for label in PARTS:
            feed_dict[self.inputs[label[1:]]] = np.asarray(inputs[label]).T
            feed_dict[self.inputs_sizes[label[1:]]] = inputs_sizes[label]
        feed_dict[self.root_time_steps] = root_time_steps
        feed_dict[self.output_time_steps] = output_time_steps
        feed_dict[self.depth] = depth
        feed_dict[self.labels_targets] = labels
        feed_dict[self.outputs_targets] = outputs
        return feed_dict

    @property
    def data_set(self) -> (list, list, list):
        data_set = list(self._data_set)
        data_set_length = len(self._data_set)
        not_allocated = data_set_length
        test_set_length = min(not_allocated, int(data_set_length * TEST_SET))
        not_allocated -= test_set_length
        train_set_length = min(not_allocated, int(data_set_length * TRAIN_SET))
        not_allocated -= train_set_length
        validation_set_length = min(not_allocated, int(data_set_length * VALIDATION_SET))
        not_allocated -= validation_set_length
        if test_set_length < MINIMUM_DATA_SET_LENGTH:
            args = (test_set_length, MINIMUM_DATA_SET_LENGTH)
            raise ValueError("Length of the test set is very small, length = %d < %d" % args)
        if train_set_length < MINIMUM_DATA_SET_LENGTH:
            args = (train_set_length, MINIMUM_DATA_SET_LENGTH)
            raise ValueError("Length of the train set is very small, length = %d < %d" % args)
        if validation_set_length < MINIMUM_DATA_SET_LENGTH:
            args = (validation_set_length, MINIMUM_DATA_SET_LENGTH)
            raise ValueError("Length of the validation set is very small, length = %d < %d" % args)
        test_set = data_set[-test_set_length:]
        data_set = data_set[:-test_set_length]
        random.shuffle(data_set)
        train_set = data_set[-train_set_length:]
        data_set = data_set[:-train_set_length]
        validation_set = data_set[-validation_set_length:]
        return train_set, validation_set, test_set

    def correct_target(self, feed_dict, outputs) -> dict:
        labels_targets = feed_dict[self.labels_targets]
        outputs_targets = feed_dict[self.outputs_targets]
        labels_targets, outputs_targets = greedy_correct(labels_targets, outputs_targets, outputs)
        # labels_targets, outputs_targets = nearest_correct(labels_target, outputs_target, outputs, 0.5)
        feed_dict[self.labels_targets] = labels_targets
        feed_dict[self.outputs_targets] = outputs_targets
        return feed_dict

    @trace
    def pretrain(self):
        pass

    @trace
    def train(self):
        try:
            formatter = Formatter(
                heads=("epoch", "time", "train", "validation"),
                formats=("d", ".4f", ".4f", ".4f"),
                sizes=(10, 20, 20, 20),
                rows=(0, 1, 2, 3),
                height=10)
            figure = ProxyFigure("train")
            validation_loss_graph = figure.fill_graph(3, 1, 1, mode="-r", color="red", alpha=0.2)
            train_loss_graph = figure.fill_graph(3, 1, 1, mode="-b", color="blue", alpha=0.2)
            false_negative_graph = figure.curve(3, 1, 2, mode="-m")
            true_negative_graph = figure.curve(3, 1, 2, mode="-y")
            false_positive_graph = figure.curve(3, 1, 2, mode="-r")
            true_positive_graph = figure.curve(3, 1, 2, mode="-g")
            accuracy_graph = figure.curve(3, 1, 3, mode="-g")
            figure.set_y_label(3, 1, 1, "loss")
            figure.set_y_label(3, 1, 2, "typed errors")
            figure.set_y_label(3, 1, 3, "accuracy")
            figure.set_x_label(3, 1, 3, "epoch")
            smoothed_false_negative = SmoothedValue(SMOOTHING)
            smoothed_false_positive = SmoothedValue(SMOOTHING)
            smoothed_true_negative = SmoothedValue(SMOOTHING)
            smoothed_true_positive = SmoothedValue(SMOOTHING)
            smoothed_accuracy = SmoothedValue(SMOOTHING)
            with tf.variable_scope("summaries"):
                false_negative_placeholder = tf.placeholder(tf.float32, shape=[], name="false_negative")
                false_positive_placeholder = tf.placeholder(tf.float32, shape=[], name="false_positive")
                true_negative_placeholder = tf.placeholder(tf.float32, shape=[], name="true_negative")
                true_positive_placeholder = tf.placeholder(tf.float32, shape=[], name="true_positive")
                accuracy_placeholder = tf.placeholder(tf.float32, shape=[], name="accuracy")
                tf.summary.scalar("false_negative", false_negative_placeholder)
                tf.summary.scalar("false_positive", false_positive_placeholder)
                tf.summary.scalar("true_negative", true_negative_placeholder)
                tf.summary.scalar("true_positive", true_positive_placeholder)
                tf.summary.scalar("accuracy", accuracy_placeholder)
            self.summaries = tf.summary.merge_all()
            config = tf.ConfigProto()
            config.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1
            with tf.Session(config=config) as session, tf.device('/cpu:0'):
                writer = tf.summary.FileWriter(ANALYSER_SUMMARIES, session.graph)
                session.run(tf.global_variables_initializer())
                for epoch in range(TRAIN_EPOCHS):
                    train_set, validation_set, test_set = self.data_set
                    start = time.time()
                    for batch in train_set:
                        feed_dict = self.build_feed_dict(batch)
                        outputs = session.run(self.outputs, feed_dict)
                        feed_dict = self.correct_target(feed_dict, outputs)
                        session.run(self.optimizer, feed_dict)
                    train_losses = []
                    for batch in train_set:
                        feed_dict = self.build_feed_dict(batch)
                        outputs = session.run(self.outputs, feed_dict)
                        feed_dict = self.correct_target(feed_dict, outputs)
                        train_losses.append(session.run(self.accuracy_loss, feed_dict))
                    validation_losses = []
                    true_positive, true_negative, false_negative, false_positive = [], [], [], []
                    for batch in validation_set:
                        feed_dict = self.build_feed_dict(batch)
                        outputs = session.run(self.outputs, feed_dict)
                        feed_dict = self.correct_target(feed_dict, outputs)
                        validation_losses.append(session.run(self.accuracy_loss, feed_dict))
                        labels, outputs = session.run((self.labels, self.outputs), feed_dict)
                        outputs = np.argmax(outputs, 3)
                        labels = np.argmax(labels, 2)
                        outputs_targets = feed_dict[self.outputs_targets]
                        labels_targets = feed_dict[self.labels_targets]
                        result = calc_accuracy(outputs_targets, outputs, labels_targets, labels)
                        true_positive.extend(result[0])
                        true_negative.extend(result[1])
                        false_negative.extend(result[2])
                        false_positive.extend(result[3])
                    stop = time.time()
                    delay = stop - start
                    train_loss = np.mean(train_losses)
                    deviation_train_loss = np.sqrt(np.var(train_losses))
                    validation_loss = np.mean(validation_losses)
                    deviation_validation_loss = np.sqrt(np.var(validation_losses))
                    formatter.print(epoch, delay, train_loss, validation_loss)
                    train_loss_graph.append(epoch, train_loss, deviation_train_loss)
                    validation_loss_graph.append(epoch, validation_loss, deviation_validation_loss)
                    number_conditions = np.sum(false_negative + false_positive + true_positive + true_negative)
                    accuracy = np.sum(true_positive + true_negative) / number_conditions
                    false_negative = np.sum(false_negative) / number_conditions
                    false_positive = np.sum(false_positive) / number_conditions
                    true_negative = np.sum(true_negative) / number_conditions
                    true_positive = np.sum(true_positive) / number_conditions
                    smoothed_accuracy(accuracy)
                    smoothed_false_negative(false_negative)
                    smoothed_false_positive(false_positive)
                    smoothed_true_negative(true_negative)
                    smoothed_true_positive(true_positive)
                    accuracy_graph.append(epoch, smoothed_accuracy())
                    false_negative_graph.append(epoch, smoothed_false_negative())
                    false_positive_graph.append(epoch, smoothed_false_positive())
                    true_negative_graph.append(epoch, smoothed_true_negative())
                    true_positive_graph.append(epoch, smoothed_true_positive())
                    summaries = session.run(self.summaries, {
                        false_negative_placeholder: false_negative,
                        false_positive_placeholder: false_positive,
                        true_positive_placeholder: true_positive,
                        true_negative_placeholder: true_negative,
                        accuracy_placeholder: accuracy
                    })
                    figure.draw()
                    figure.save(self.folder_path + "/train.png")
                    writer.add_summary(summaries)
                    writer.flush()
                    if np.isnan(train_loss) or np.isnan(validation_loss):
                        raise Net.NaNException()
                    self.save(session)
        except Net.NaNException as ex:
            logging.info(ex)
        finally:
            writer.close()
            figure.save(self.folder_path + "/train.png")
            ProxyFigure.destroy()

    @trace
    def test(self, model_path: str = None):
        formatter = Formatter(
            heads=("loss", "target", *(["output", "prob"] * TOP)),
            formats=(".4f", "s", *(["s", ".4f"] * TOP)),
            sizes=(12, 20, *([20, 12] * TOP)),
            rows=range(2 + 2 * TOP))
        config = tf.ConfigProto()
        config.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1
        with tf.Session(config=config) as session, tf.device('/cpu:0'):
            session.run(tf.global_variables_initializer())
            self.restore(session, model_path)
            writer = tf.summary.FileWriter(ANALYSER_SUMMARIES, session.graph)
            writer.add_summary(session.run(self.summaries))
            writer.flush()
            writer.close()
            train_set, validation_set, test_set = self.data_set
            for batch in test_set:
                feed_dict = self.build_feed_dict(batch)
                outputs = session.run(self.outputs, feed_dict)
                feed_dict = self.correct_target(feed_dict, outputs)
                fetches = (
                    self.accuracy_loss, self.labels_targets, self.top_labels, self.outputs_targets, self.top_outputs,
                    self.inputs)
                ls, lts, tls, ots, tos, inps = session.run(fetches, feed_dict)
                tlis = tls.indices
                tlps = tls.values
                tois = tos.indices
                tops = tos.values
                inps = [inps[label[1:]] for label in PARTS]
                for l, lt, tli, tlp, ot, toi, top, *inp in zip(ls, lts, tlis, tlps, ots, tois, tops, *inps):
                    formatter.print_head()
                    for lt_i, tli_i, tlp_i, ot_i, toi_i, top_i in zip(lt, tli, tlp, ot, toi, top):
                        lt_i = embeddings.labels().get_name(int(lt_i))
                        tli_i = (embeddings.labels().get_name(int(i)) for i in tli_i)
                        args = (elem for pair in zip(tli_i, tlp_i) for elem in pair)
                        reminder = ["", 0.0] * (TOP - 2)
                        formatter.print(l, lt_i, *args, *reminder)
                        for ot_ij, toi_ij, top_ij in zip(ot_i, toi_i, top_i):
                            ot_ij = embeddings.tokens().get_name(int(ot_ij))
                            toi_ij = (embeddings.tokens().get_name(int(i)) for i in toi_ij)
                            args = (elem for pair in zip(toi_ij, top_ij) for elem in pair)
                            formatter.print(l, ot_ij, *args)
                        formatter.print_delimiter()
                    for inp_i, label in zip(inp, PARTS):
                        line = " ".join(embeddings.words().get_name(int(i)) for i in inp_i)
                        lines = (line.strip() for line in line.replace(PAD, " ").split(NEXT))
                        text = "\n".join(line for line in lines if len(line) > 0)
                        formatter.print_appendix(text, label)
                    formatter.print_lower_delimiter()

    @trace
    def test1(self):
        formatter = Formatter(
            heads=(["output", "prob"] * TOP),
            formats=(["s", ".4f"] * TOP),
            sizes=([20, 12] * TOP),
            rows=range(2 * TOP))
        config = tf.ConfigProto()
        config.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1
        with tf.Session(config=config) as session, tf.device('/cpu:0'):
            session.run(tf.global_variables_initializer())
            self.restore(session)
            feed_dict = {}
            for label in PARTS:
                the = embeddings.words().get_index("the")
                may = embeddings.words().get_index("may")
                be = embeddings.words().get_index("be")
                pad = embeddings.words().get_index(PAD)
                feed_dict[self.inputs[label[1:]]] = np.asarray([[the, may, be] * 4 + [pad] * 4] * BATCH_SIZE)
                feed_dict[self.inputs_sizes[label[1:]]] = [11] * BATCH_SIZE
            feed_dict[self.root_time_steps] = 10
            feed_dict[self.output_time_steps] = 0
            feed_dict[self.depth] = 3
            fetches = (self.top_labels, self.top_outputs)
            top_labels, top_outputs = session.run(fetches, feed_dict)
            lis = top_labels.indices
            lps = top_labels.values
            ois = top_outputs.indices
            ops = top_outputs.values
            for li, lp, oi, op in zip(lis, lps, ois, ops):
                formatter.print_head()
                for not_first, (li_i, lp_i, oi_i, op_i) in enumerate(zip(li, lp, oi, op)):
                    if not_first:
                        formatter.print_delimiter()
                    li_i = (embeddings.labels().get_name(int(i)) for i in li_i)
                    formatter.print(*(elem for pair in zip(li_i, lp_i) for elem in pair), "", 0.0)
                    for oi_ij, op_ij in zip(oi_i, op_i):
                        oi_ij = (embeddings.tokens().get_name(int(i)) for i in oi_ij)
                        formatter.print(*(elem for pair in zip(oi_ij, op_ij) for elem in pair))
                formatter.print_lower_delimiter()


if __name__ == '__main__':
    init()
    AnalyserNet().test1()
