import logging
import os
import random
import time

from live_plotter.proxy.ProxyFigure import ProxyFigure
from tensorflow.python.ops.rnn_cell_impl import GRUCell

from config import init
from constants.analyser import *
from constants.paths import ANALYSER, ANALYSER_METHODS, ANALYSER_SUMMARIES, ANALYSER_GRAPH
from constants.tags import PAD
from seq2seq.Net import Net
from seq2seq.analyser_rnn import *
from seq2seq.utils import *
from seq2seq.utils import cross_entropy_loss, l2_loss
from utils import dumpers
from utils.Formatter import Formatter
from utils.wrappers import trace


class AnalyserNet(Net):
    @trace
    def __init__(self):
        super().__init__("analyser", ANALYSER)
        with vs.variable_scope(self.name):
            self.docs = {}
            self.embeddings = {}
            self.docs_sizes = {}
            cells_fw = {}
            cells_bw = {}
            for label in PARTS:
                label = label[1:]
                indexes = tf.placeholder(tf.int32, [BATCH_SIZE, None], "indexes_%s" % label)
                self.embeddings[label] = tf.gather(tf.constant(np.asarray(embeddings.words().idx2emb)), indexes)
                self.docs[label] = indexes
                self.docs_sizes[label] = tf.placeholder(tf.int32, [BATCH_SIZE], "input_sizes_%s" % label)
                cells_fw[label] = [GRUCell(INPUTS_STATE_SIZE) for _ in range(NUM_ENCODERS)]
                cells_bw[label] = [GRUCell(INPUTS_STATE_SIZE) for _ in range(NUM_ENCODERS)]
            conditions_cells = [GRUCell(CONDITIONS_STATE_SIZE) for _ in range(NUM_DECODERS)]
            sequence_cells = [GRUCell(SEQUENCE_STATE_SIZE) for _ in range(NUM_DECODERS)]
            strings_cell = GRUCell(STRINGS_STATE_SIZE)
            self.num_conditions = tf.placeholder(tf.int32, [], "num_conditions")
            self.sequence_length = tf.placeholder(tf.int32, [], "sequence_length")
            self.string_length = tf.placeholder(tf.int32, [], "string_length")
            self.labels_targets = tf.placeholder(tf.int32, [BATCH_SIZE, None], "labels_targets")
            self.tokens_targets = tf.placeholder(tf.int32, [BATCH_SIZE, None, None], "outputs_targets")
            self.strings_targets = tf.placeholder(tf.int32, [BATCH_SIZE, None, None, None], "strings_target")
            self.strings_mask = tf.placeholder(tf.float32, [BATCH_SIZE, None, None], "strings_target_mask")
            self.tree_depth = tf.placeholder(tf.int32, [], "depth")
            projection = input_projection(self.embeddings, INPUT_SIZE, tf.float32)
            inputs_states = sequence_input(cells_bw, cells_fw, projection, self.docs_sizes, tf.float32)
            if os.environ['OUTPUT_TYPE'] == "tree":
                outputs, tokens_states, self.attentions_masks = tree_output(
                    inputs_states, conditions_cells, self.num_conditions, CONDITIONS_NUM_HEADS,
                    self.tree_depth,
                    NUM_LABELS, NUM_TOKENS, tf.float32)
            elif os.environ['OUTPUT_TYPE'] in ("bfs_sequence", "dfs_sequence"):
                outputs, tokens_states, self.attentions_masks = sequence_output(
                    inputs_states, conditions_cells, self.num_conditions, CONDITIONS_NUM_HEADS,
                    sequence_cells, self.sequence_length, SEQUENCE_NUM_HEADS,
                    NUM_LABELS, NUM_TOKENS, tf.float32)
            self.labels_logits, self.raw_labels, self.tokens_logits, self.raw_tokens = outputs
            self.strings_logits, self.raw_strings = strings_output(
                strings_cell, inputs_states, tokens_states, self.string_length, NUM_WORDS, STRINGS_HIDDEN_SIZE)
            self.top_labels = tf.nn.top_k(self.raw_labels, min(2, TOP))
            self.top_tokens = tf.nn.top_k(self.raw_tokens, TOP)
            self.labels = tf.argmax(self.raw_labels, 2)
            self.tokens = tf.argmax(self.raw_tokens, 3)
            self.strings = tf.argmax(self.raw_strings, 4)
            self.scope = vs.get_variable_scope().name
            self.loss, self.complex_loss = self.build_loss()
        self.optimizer = tf.train.AdamOptimizer().minimize(self.complex_loss)
        self._data_set = dumpers.pkl_load(ANALYSER_METHODS)
        self.add_variable_summaries()
        self.summaries = tf.summary.merge_all()

    def add_variable_summaries(self):
        tf.summary.histogram("sum", tf.concat([tf.reshape(variable, [-1]) for variable in self.variables], 0))
        for variable in self.variables:
            tf.summary.histogram(variable.name, variable)

    def build_loss(self):
        with vs.variable_scope("loss"):
            # labels_loss = cross_entropy_loss(self.labels_targets, self.labels_logits)
            outputs_loss = cross_entropy_loss(self.tokens_targets, self.tokens_logits)
            mask = tf.expand_dims(tf.expand_dims(self.strings_mask, -1), -1)
            strings_loss = cross_entropy_loss(self.strings_targets, mask * self.strings_logits)
            _l2_loss = L2_LOSS_WEIGHT * l2_loss(self.variables)
            loss = outputs_loss + strings_loss  # + labels_loss
            complex_loss = loss + _l2_loss
        return loss, complex_loss

    def build_feed_dict(self, batch) -> dict:
        inputs, outputs, parameters = batch
        docs, docs_sizes = inputs
        labels, tokens, strings, strings_mask = outputs
        num_conditions, sequence_length, string_length, tree_depth = parameters
        feed_dict = {}
        for label in PARTS:
            feed_dict[self.docs[label[1:]]] = np.asarray(docs[label]).T
            feed_dict[self.docs_sizes[label[1:]]] = docs_sizes[label]
        feed_dict[self.num_conditions] = num_conditions
        feed_dict[self.sequence_length] = sequence_length
        feed_dict[self.string_length] = string_length
        feed_dict[self.tree_depth] = tree_depth
        feed_dict[self.labels_targets] = labels
        feed_dict[self.tokens_targets] = tokens
        feed_dict[self.strings_targets] = strings
        feed_dict[self.strings_mask] = strings_mask
        return feed_dict

    @property
    def data_set(self) -> (list, list, list):
        data_set = list(self._data_set)
        data_set_length = len(data_set)
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
        tokens_targets = feed_dict[self.tokens_targets]
        strings_targets = feed_dict[self.strings_targets]
        strings_mask = feed_dict[self.strings_mask]
        targets = (labels_targets, tokens_targets, strings_targets, strings_mask)
        labels_targets, tokens_targets, strings_targets, strings_mask = greedy_correct(targets, outputs)
        # labels_targets, tokens_targets, strings_targets, strings_mask = nearest_correct(targets, outputs, 0.5)
        feed_dict[self.labels_targets] = labels_targets
        feed_dict[self.tokens_targets] = tokens_targets
        feed_dict[self.strings_targets] = strings_targets
        feed_dict[self.strings_mask] = strings_mask
        return feed_dict

    @trace
    def pretrain(self):
        pass

    @trace
    def train(self):
        try:
            heads = ("epoch", "time", "train_accuracy", "train_loss", "validation_accuracy", "validation_loss")
            formats = ("d", ".4f", ".4f", ".4f", ".4f", ".4f")
            formatter = Formatter(heads, formats, (10, 20, 20, 20, 21, 20), range(6), 10)
            figure = ProxyFigure("train")
            validation_loss_graph = figure.distributed_curve(3, 1, 1, mode="-r", color="red", alpha=0.2)
            train_loss_graph = figure.distributed_curve(3, 1, 1, mode="-b", color="blue", alpha=0.2)
            smoothed_train_accuracy_graph = figure.smoothed_curve(3, 1, 3, 0.64, mode="-b")
            smoothed_validation_accuracy_graph = figure.smoothed_curve(3, 1, 3, 0.64, mode="-r")
            smoothed_false_negative_graph = figure.smoothed_curve(3, 1, 2, 0.64, mode="-m")
            smoothed_true_negative_graph = figure.smoothed_curve(3, 1, 2, 0.64, mode="-y")
            smoothed_false_positive_graph = figure.smoothed_curve(3, 1, 2, 0.64, mode="-r")
            smoothed_true_positive_graph = figure.smoothed_curve(3, 1, 2, 0.64, mode="-g")
            figure.set_y_label(3, 1, 1, "loss")
            figure.set_y_label(3, 1, 2, "error")
            figure.set_y_label(3, 1, 3, "accuracy")
            figure.set_x_label(3, 1, 3, "epoch")
            figure.set_label(3, 1, 1, "Train and validation losses")
            figure.set_label(3, 1, 2, "Validation typed errors")
            figure.set_label(3, 1, 3, "Train and validation accuracies")
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
                        fetches = (self.raw_labels, self.raw_tokens, self.raw_strings)
                        outputs = session.run(fetches, feed_dict)
                        feed_dict = self.correct_target(feed_dict, outputs)
                        session.run(self.optimizer, feed_dict)
                    train_losses = []
                    train_true_positive, train_true_negative = [], []
                    train_false_negative, train_false_positive = [], []
                    for batch in train_set:
                        feed_dict = self.build_feed_dict(batch)
                        fetches = (self.raw_labels, self.raw_tokens, self.raw_strings)
                        outputs = session.run(fetches, feed_dict)
                        feed_dict = self.correct_target(feed_dict, outputs)
                        train_losses.append(session.run(self.loss, feed_dict))
                        labels, tokens = session.run((self.raw_labels, self.raw_tokens), feed_dict)
                        tokens = np.argmax(tokens, 3)
                        outputs_targets = feed_dict[self.tokens_targets]
                        # labels = np.argmax(labels, 2)
                        # labels_targets = feed_dict[self.labels_targets]
                        # result = calc_accuracy(outputs_targets, outputs, labels_targets, labels)
                        errors = calc_accuracy(outputs_targets, tokens)
                        train_true_positive.extend(errors[0])
                        train_true_negative.extend(errors[1])
                        train_false_negative.extend(errors[2])
                        train_false_positive.extend(errors[3])
                    validation_losses = []
                    validation_true_positive, validation_true_negative = [], []
                    validation_false_negative, validation_false_positive = [], []
                    for batch in validation_set:
                        feed_dict = self.build_feed_dict(batch)
                        fetches = (self.raw_labels, self.raw_tokens, self.raw_strings)
                        outputs = session.run(fetches, feed_dict)
                        feed_dict = self.correct_target(feed_dict, outputs)
                        validation_losses.append(session.run(self.loss, feed_dict))
                        labels, tokens = session.run((self.raw_labels, self.raw_tokens), feed_dict)
                        tokens = np.argmax(tokens, 3)
                        outputs_targets = feed_dict[self.tokens_targets]
                        # labels = np.argmax(labels, 2)
                        # labels_targets = feed_dict[self.labels_targets]
                        # result = calc_accuracy(outputs_targets, outputs, labels_targets, labels)
                        errors = calc_accuracy(outputs_targets, tokens)
                        validation_true_positive.extend(errors[0])
                        validation_true_negative.extend(errors[1])
                        validation_false_negative.extend(errors[2])
                        validation_false_positive.extend(errors[3])
                    stop = time.time()
                    delay = stop - start
                    train_loss = np.mean(train_losses)
                    deviation_train_loss = np.sqrt(np.var(train_losses))
                    validation_loss = np.mean(validation_losses)
                    deviation_validation_loss = np.sqrt(np.var(validation_losses))
                    train_loss_graph.append(epoch, train_loss, deviation_train_loss)
                    validation_loss_graph.append(epoch, validation_loss, deviation_validation_loss)
                    validation_number_conditions = np.sum(validation_false_negative + validation_false_positive +
                                                          validation_true_positive + validation_true_negative)
                    validation_false_negative = np.sum(validation_false_negative) / validation_number_conditions
                    validation_false_positive = np.sum(validation_false_positive) / validation_number_conditions
                    validation_true_negative = np.sum(validation_true_negative) / validation_number_conditions
                    validation_true_positive = np.sum(validation_true_positive) / validation_number_conditions
                    validation_accuracy = validation_true_positive + validation_true_negative
                    train_number_conditions = np.sum(train_false_negative + train_false_positive +
                                                     train_true_positive + train_true_negative)
                    train_accuracy = np.sum(train_true_positive + train_true_negative) / train_number_conditions
                    smoothed_train_accuracy_graph.append(epoch, train_accuracy)
                    smoothed_validation_accuracy_graph.append(epoch, validation_accuracy)
                    smoothed_false_negative_graph.append(epoch, validation_false_negative)
                    smoothed_false_positive_graph.append(epoch, validation_false_positive)
                    smoothed_true_negative_graph.append(epoch, validation_true_negative)
                    smoothed_true_positive_graph.append(epoch, validation_true_positive)
                    formatter.print(epoch, delay, train_accuracy, train_loss, validation_accuracy, validation_loss)
                    figure.draw()
                    figure.save(ANALYSER_GRAPH)
                    writer.add_summary(session.run(self.summaries))
                    writer.flush()
                    if np.isnan(train_loss) or np.isnan(validation_loss):
                        raise Net.NaNException()
                    self.save(session)
        except Net.NaNException as ex:
            logging.info(ex)
        finally:
            writer.close()
            figure.save(ANALYSER_GRAPH)
            ProxyFigure.destroy()

    @trace
    def test(self, model_path: str = None):
        formatter = Formatter(heads=("loss", "accuracy", "target", *(["output", "prob"] * TOP)),
                              formats=(".4f", ".4f", "s", *(["s", ".4f"] * TOP)),
                              sizes=(12, 12, 15, *([15, 12] * TOP)),
                              rows=range(3 + 2 * TOP))
        appendix_formatter = Formatter(heads=("label", "text"),
                                       formats=("s", "s"),
                                       sizes=(12, 12 + 15 + sum([15, 12] * TOP) + TOP * 2 + 1),
                                       rows=(0, 1))
        config = tf.ConfigProto()
        config.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1
        with tf.Session(config=config) as session, tf.device('/cpu:0'):
            session.run(tf.global_variables_initializer())
            self.restore(session, model_path)
            train_set, validation_set, test_set = self.data_set
            for batch in test_set:
                feed_dict = self.build_feed_dict(batch)
                fetches = (self.raw_labels, self.raw_tokens, self.raw_strings)
                outputs = session.run(fetches, feed_dict)
                feed_dict = self.correct_target(feed_dict, outputs)
                fetches = (self.num_conditions, self.sequence_length, self.labels_targets, self.tokens_targets)
                root_time_steps, output_time_steps, labels_targets, outputs_targets = session.run(fetches, feed_dict)
                fetches = (self.loss, self.top_labels, self.top_tokens, self.docs, self.attentions_masks)
                losses, top_labels, top_outputs, inputs, mask = session.run(fetches, feed_dict)
                tokens, strings = session.run((self.tokens, self.strings), feed_dict)
                inputs = [list(inputs[label[1:]]) for label in PARTS]
                inputs = np.asarray(inputs)
                shape_length = len(inputs.shape)
                inputs = inputs.transpose([1, 0, *range(2, shape_length)])
                mask = transpose_mask(mask, CONDITIONS_NUM_HEADS)
                top_labels_indices = top_labels.indices
                top_labels_probabilities = top_labels.values
                top_outputs_indices = top_outputs.indices
                top_outputs_probabilities = top_outputs.values
                errors = calc_accuracy(outputs_targets, tokens)
                true_positive, true_negative, false_negative, false_positive = errors
                for i in range(BATCH_SIZE):
                    num_conditions = true_positive[i] + true_negative[i] + false_negative[i] + false_positive[i]
                    accuracy = (true_positive[i] + true_negative[i]) / num_conditions
                    loss = losses[i]
                    formatter.print_head()
                    for j in range(root_time_steps):
                        label_target = labels_targets[i][j]
                        _top_labels_indices = top_labels_indices[i][j]
                        _top_labels_probabilities = top_labels_probabilities[i][j]
                        label_target = embeddings.labels().get_name(label_target)
                        _top_labels_indices = (embeddings.labels().get_name(i) for i in _top_labels_indices)
                        labels = (e for p in zip(_top_labels_indices, _top_labels_probabilities) for e in p)
                        reminder = ["", 0.0] * (TOP - 2)
                        formatter.print(loss, accuracy, label_target, *labels, *reminder)
                        for k in range(output_time_steps):
                            output_target = outputs_targets[i][j][k]
                            _top_outputs_indices = top_outputs_indices[i][j][k]
                            _top_outputs_probabilities = top_outputs_probabilities[i][j][k]
                            output_target = embeddings.tokens().get_name(output_target)
                            _top_outputs_indices = (embeddings.tokens().get_name(i) for i in _top_outputs_indices)
                            outputs = (e for p in zip(_top_outputs_indices, _top_outputs_probabilities) for e in p)
                            formatter.print(loss, accuracy, output_target, *outputs)
                        appendix_formatter.print_delimiter()
                        for token, string in zip(tokens[i][j], strings[i][j]):
                            token = embeddings.tokens().get_name(token)
                            string = " ".join(embeddings.words().get_name(word) for word in string)
                            appendix_formatter.print(token, string)
                        appendix_formatter.print_delimiter()
                        print_doc(appendix_formatter, inputs[i], mask[i][j])
                        if j < root_time_steps - 1:
                            appendix_formatter.print_delimiter()
                        else:
                            appendix_formatter.print_lower_delimiter()

    @trace
    def test1(self):
        formatter = Formatter(["output", "prob"] * TOP, ["s", ".4f"] * TOP, [20, 12] * TOP, range(2 * TOP))
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
                feed_dict[self.docs[label[1:]]] = np.asarray([[the, may, be] * 4 + [pad] * 4] * BATCH_SIZE)
                feed_dict[self.docs_sizes[label[1:]]] = [11] * BATCH_SIZE
            feed_dict[self.num_conditions] = 10
            feed_dict[self.sequence_length] = 0
            feed_dict[self.tree_depth] = 3
            fetches = (self.top_labels, self.top_tokens)
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
                    li_i = (embeddings.labels().get_name(i) for i in li_i)
                    formatter.print(*(elem for pair in zip(li_i, lp_i) for elem in pair), "", 0.0)
                    for oi_ij, op_ij in zip(oi_i, op_i):
                        oi_ij = (embeddings.tokens().get_name(i) for i in oi_ij)
                        formatter.print(*(elem for pair in zip(oi_ij, op_ij) for elem in pair))
                formatter.print_lower_delimiter()


if __name__ == '__main__':
    init()
    AnalyserNet().test1()
