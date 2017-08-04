import random
import time

from live_plotter.proxy.ProxyFigure import ProxyFigure
from tensorflow.python.ops.rnn_cell_impl import GRUCell

from configurations.constants import *
from configurations.logger import info_logger
from configurations.paths import ANALYSER, ANALYSER_SUMMARIES
from configurations.tags import PAD
from seq2seq.Net import Net
from seq2seq.analyser_rnn import *
from seq2seq.misc import *
from utils.Formatter import Formatter
from utils.SummaryWriter import SummaryWriter
from utils.wrappers import trace, Timer


class AnalyserNet(Net):
    @trace("BUILD NET")
    def __init__(self, data_set):
        super().__init__("Analyser", ANALYSER)
        num_words = len(Embeddings.words())
        num_tokens = len(Embeddings.tokens())
        with tf.variable_scope("Input"), Timer("BUILD INPUT"):
            self.inputs = tf.placeholder(tf.int32, [BATCH_SIZE, None], "inputs")
            self.inputs_length = tf.placeholder(tf.int32, [BATCH_SIZE], "inputs_length")
            self.num_conditions = tf.placeholder(tf.int32, [], "num_conditions")
            self.string_length = tf.placeholder(tf.int32, [], "string_length")
            self.tree_height = tf.placeholder(tf.int32, [], "depth")
            self.num_tokens = tf.placeholder(tf.int32, [], "num_tokens")
            self.tokens_targets = tf.placeholder(tf.int32, [BATCH_SIZE, None, None], "outputs_targets")
            self.strings_targets = tf.placeholder(tf.int32, [BATCH_SIZE, None, None, None], "strings_target")
            self.W_strings = tf.placeholder(tf.float32, [BATCH_SIZE, None, None, 1, 1], "W_strings")
            self.B_strings = tf.placeholder(tf.float32, [BATCH_SIZE, None, None, None, num_words], "B_strings")
        with tf.variable_scope(self.name), Timer("BUILD BODY"):
            input_cell = (GRUCell(INPUTS_STATE_SIZE), GRUCell(INPUTS_STATE_SIZE))
            tree_output_cell = GRUCell(TREE_OUTPUT_STATE_SIZE)
            string_output_cell = GRUCell(STRING_OUTPUT_STATE_SIZE)
            _embeddings = tf.gather(tf.constant(np.asarray(Embeddings.words().idx2emb)), self.inputs)
            attention_states = sequence_input(*input_cell, _embeddings, self.inputs_length, INPUT_HIDDEN_SIZE)
            if OUTPUT_TYPE == "tree":
                self.tokens_logits, self.raw_tokens, states, self.attention = tree_output(
                    tree_output_cell, attention_states, self.num_conditions, self.tree_height, num_tokens)
            else:
                raise ValueError("Output type '%s' hasn't expected" % OUTPUT_TYPE)
            self.strings_logits, self.raw_strings = string_output(
                string_output_cell, attention_states, states, self.string_length, num_words)
            self.scope = vs.get_variable_scope().name
        with tf.variable_scope("Output"), Timer("BUILD OUTPUT"):
            self.top_tokens = tf.nn.top_k(self.raw_tokens, TOP)
            self.tokens = tf.argmax(self.raw_tokens, 3)
            self.strings = tf.argmax(self.raw_strings, 4)
        with tf.variable_scope("Loss"), Timer("BUILD LOSS"):
            nop = Embeddings.tokens().get_index(NOP)
            pad = Embeddings.words().get_index(PAD)
            self.tokens_loss = cross_entropy_loss(self.tokens_targets, self.tokens_logits, nop)
            self.strings_loss = cross_entropy_loss(self.strings_targets, self.strings_logits, pad)
            self.l2_loss = L2_LOSS_WEIGHT * l2_loss(self.variables)
            self.loss = self.tokens_loss + self.strings_loss + self.l2_loss
        with tf.variable_scope("Optimizer"), Timer("BUILD OPTIMISER"):
            self.optimizer = tf.train.AdamOptimizer().minimize(self.loss)
        with tf.variable_scope("Summaries"), Timer("BUILD SUMMARIES"):
            self.summaries = self.add_variable_summaries()
            self._data_set = data_set

    def add_variable_summaries(self):
        variables = [tf.reshape(variable, [-1]) for variable in self.variables]
        tf.summary.histogram("Summary", tf.concat(variables, 0))
        for variable in self.variables:
            tf.summary.histogram(variable.name, variable)
        return tf.summary.merge_all()

    def build_feed_dict(self, batch) -> dict:
        inputs, outputs, parameters = batch
        inputs, inputs_length = inputs
        tokens, strings = outputs
        num_conditions, num_tokens, string_length, tree_depth = parameters
        feed_dict = {self.inputs: inputs,
                     self.inputs_length: inputs_length,
                     self.num_conditions: num_conditions,
                     self.num_tokens: num_tokens,
                     self.string_length: string_length,
                     self.tree_height: tree_depth,
                     self.tokens_targets: tokens,
                     self.strings_targets: strings}
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
        targets = feed_dict[self.tokens_targets]
        strings_targets = feed_dict[self.strings_targets]
        dependencies = (strings_targets,)
        targets, dependencies = greedy_correct(targets, outputs, dependencies)
        # targets, dependencies = nearest_correct(targets, outputs, dependencies, outputs, 0.5)
        strings_targets = dependencies[0]
        feed_dict[self.tokens_targets] = targets
        feed_dict[self.strings_targets] = strings_targets
        return feed_dict

    @trace("TRAIN NET")
    def train(self):
        heads = ("epoch", "time", "train_accuracy", "train_loss", "validation_accuracy", "validation_loss")
        formats = ("d", ".4f", ".4f", ".4f", ".4f", ".4f")
        formatter = Formatter(heads, formats, (9, 20, 20, 20, 21, 21), range(6), 10)
        figure = ProxyFigure("train", self.folder_path + "/train.png")
        validation_loss_graph = figure.curve(3, 1, 1, mode="-r")
        train_loss_graph = figure.curve(3, 1, 1, mode="-b")
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
        session = tf.Session(config=config)
        device = tf.device('/cpu:0')
        writer = SummaryWriter(ANALYSER_SUMMARIES, session, self.summaries)
        with session, device, writer, figure:
            session.run(tf.global_variables_initializer())
            for epoch in range(TRAIN_EPOCHS):
                train_set, validation_set, test_set = self.data_set
                start = time.time()
                for batch in train_set:
                    feed_dict = self.build_feed_dict(batch)
                    raw_tokens = session.run(self.raw_tokens, feed_dict)
                    feed_dict = self.correct_target(feed_dict, raw_tokens)
                    session.run(self.optimizer, feed_dict)
                trains_loss, trains_accuracy = [], []
                trains_true_positive, trains_true_negative = [], []
                trains_false_negative, trains_false_positive = [], []
                for batch in train_set:
                    feed_dict = self.build_feed_dict(batch)
                    raw_tokens = session.run(self.raw_tokens, feed_dict)
                    feed_dict = self.correct_target(feed_dict, raw_tokens)
                    tokens, loss = session.run((self.tokens, self.loss), feed_dict)
                    accuracy, errors = batch_accuracy(feed_dict[self.tokens_targets], tokens)
                    trains_true_positive.extend(errors[0])
                    trains_true_negative.extend(errors[1])
                    trains_false_negative.extend(errors[2])
                    trains_false_positive.extend(errors[3])
                    trains_accuracy.append(accuracy)
                    trains_loss.append(loss)
                validations_loss, validations_accuracy = [], []
                validations_true_positive, validations_true_negative = [], []
                validations_false_negative, validations_false_positive = [], []
                for batch in validation_set:
                    feed_dict = self.build_feed_dict(batch)
                    raw_tokens = session.run(self.raw_tokens, feed_dict)
                    feed_dict = self.correct_target(feed_dict, raw_tokens)
                    tokens, loss = session.run((self.tokens, self.loss), feed_dict)
                    accuracy, errors = batch_accuracy(feed_dict[self.tokens_targets], tokens)
                    validations_true_positive.extend(errors[0])
                    validations_true_negative.extend(errors[1])
                    validations_false_negative.extend(errors[2])
                    validations_false_positive.extend(errors[3])
                    validations_accuracy.append(accuracy)
                    validations_loss.append(loss)
                stop = time.time()
                delay = stop - start
                train_loss = np.mean(trains_loss)
                validation_loss = np.mean(validations_loss)
                train_accuracy = np.mean(trains_accuracy)
                validation_accuracy = np.mean(validations_accuracy)
                validation_false_negative = np.mean(validations_false_negative)
                validation_false_positive = np.mean(validations_false_positive)
                validation_true_negative = np.mean(validations_true_negative)
                validation_true_positive = np.mean(validations_true_positive)
                train_loss_graph.append(epoch, train_loss)
                validation_loss_graph.append(epoch, validation_loss)
                smoothed_train_accuracy_graph.append(epoch, train_accuracy)
                smoothed_validation_accuracy_graph.append(epoch, validation_accuracy)
                smoothed_false_negative_graph.append(epoch, validation_false_negative)
                smoothed_false_positive_graph.append(epoch, validation_false_positive)
                smoothed_true_negative_graph.append(epoch, validation_true_negative)
                smoothed_true_positive_graph.append(epoch, validation_true_positive)
                formatter.print(epoch, delay, train_accuracy, train_loss, validation_accuracy, validation_loss)
                if np.isnan(train_loss) or np.isnan(validation_loss):
                    info_logger.info("NaN detected")
                    break
                figure.draw()
                figure.save()
                writer.update()
                self.save(session)

    @trace("TEST NET")
    def test(self, model_path: str = None):
        heads = ("accuracy", "tokens loss", "strings loss", "target", *(["output", "prob"] * TOP))
        formats = (*([".4f"] * 3), "s", *(["s", ".4f"] * TOP))
        sizes = (*([15] * 3), *([13] * (1 + 2 * TOP)))
        formatter = Formatter(heads, formats, sizes, range(4 + 2 * TOP))
        heads = ("label", "text")
        formats = ("s", "s")
        sizes = (formatter.row_size(0), formatter.size - formatter.row_size(0) - 3)
        formatter0 = Formatter(heads, formats, sizes, (0, 1))
        formatter1 = Formatter(["text"], ["s"], [formatter.size - 2], [0])
        config = tf.ConfigProto()
        config.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1
        with tf.Session(config=config) as session, tf.device('/cpu:0'):
            session.run(tf.global_variables_initializer())
            self.restore(session, model_path)
            train_set, validation_set, test_set = self.data_set
            for batch in test_set:
                feed_dict = self.build_feed_dict(batch)
                raw_tokens = session.run(self.raw_tokens, feed_dict)
                feed_dict = self.correct_target(feed_dict, raw_tokens)
                fetches = (self.top_tokens, self.raw_tokens, self.tokens, self.tokens_loss)
                top_tokens, raw_tokens, tokens, tokens_loss = session.run(fetches, feed_dict)
                fetches = (self.attention, self.strings, self.strings_loss)
                attention, strings, strings_loss = session.run(fetches, feed_dict)
                inputs = feed_dict[self.inputs]
                num_conditions = feed_dict[self.num_conditions]
                num_tokens = feed_dict[self.num_tokens]
                tokens_targets = feed_dict[self.tokens_targets]
                strings_targets = feed_dict[self.strings_targets]
                attention = transpose_attention(attention)
                accuracy, errors = batch_accuracy(tokens_targets, tokens)
                for i in range(BATCH_SIZE):
                    formatter.print_head()
                    for j in range(num_conditions):
                        for k in range(num_tokens):
                            token_target = tokens_targets[i][j][k]
                            top_tokens_indices = top_tokens.indices[i][j][k]
                            top_tokens_probabilities = top_tokens.values[i][j][k]
                            token_target = Embeddings.tokens().get_name(token_target)
                            top_tokens_indices = (Embeddings.tokens().get_name(i) for i in top_tokens_indices)
                            outputs = (e for p in zip(top_tokens_indices, top_tokens_probabilities) for e in p)
                            formatter.print(accuracy[i], tokens_loss[i], strings_loss[i], token_target, *outputs)
                        formatter0.print_delimiter()
                        print_strings(formatter0, tokens[i][j], strings[i][j], strings_targets[i][j])
                        formatter0.print_delimiter()
                        print_raw_tokens(formatter0, raw_tokens[i][j])
                        formatter1.print_delimiter()
                        print_doc(formatter1, inputs[i], attention[i][j])
                        if j < num_conditions - 1:
                            formatter1.print_delimiter()
                        else:
                            formatter1.print_lower_delimiter()
