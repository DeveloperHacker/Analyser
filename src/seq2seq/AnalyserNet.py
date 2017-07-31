import logging
import random
import time

from live_plotter.proxy.ProxyFigure import ProxyFigure
from tensorflow.python.ops.rnn_cell_impl import GRUCell

from configurations.constants import *
from configurations.paths import ANALYSER, ANALYSER_DATA_SET, ANALYSER_SUMMARIES
from seq2seq.Net import Net
from seq2seq.analyser_rnn import *
from seq2seq.misc import *
from utils import dumpers
from utils.Formatter import Formatter


class AnalyserNet(Net):
    def __init__(self):
        super().__init__("Analyser", ANALYSER)
        num_words = len(Embeddings.words())
        num_tokens = len(Embeddings.tokens())
        with tf.variable_scope("Input"):
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
        with tf.variable_scope(self.name):
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
        with tf.variable_scope("Output"):
            self.top_tokens = tf.nn.top_k(self.raw_tokens, TOP)
            self.tokens = tf.argmax(self.raw_tokens, 3)
            self.strings = tf.argmax(self.raw_strings, 4)
        with tf.variable_scope("Loss"):
            self.tokens_loss = cross_entropy_loss(self.tokens_targets, self.tokens_logits)
            logits = self.W_strings * self.strings_logits + self.B_strings
            self.strings_loss = cross_entropy_loss(self.strings_targets, logits)
            self.l2_loss = L2_LOSS_WEIGHT * l2_loss(self.variables)
            self.loss = self.tokens_loss + self.strings_loss + self.l2_loss
        with tf.variable_scope("Optimizer"):
            self.optimizer = tf.train.AdamOptimizer().minimize(self.loss)
        with tf.variable_scope("Summaries"):
            self.summaries = self.add_variable_summaries()
            self._data_set = dumpers.pkl_load(ANALYSER_DATA_SET)

    def add_variable_summaries(self):
        variables = [tf.reshape(variable, [-1]) for variable in self.variables]
        tf.summary.histogram("Summary", tf.concat(variables, 0))
        for variable in self.variables:
            tf.summary.histogram(variable.name, variable)
        return tf.summary.merge_all()

    def build_feed_dict(self, batch) -> dict:
        inputs, outputs, parameters = batch
        inputs, inputs_length = inputs
        labels, tokens, strings, W_strings, B_strings = outputs
        num_conditions, num_tokens, string_length, tree_depth = parameters
        feed_dict = {self.inputs: inputs,
                     self.inputs_length: inputs_length,
                     self.num_conditions: num_conditions,
                     self.num_tokens: num_tokens,
                     self.string_length: string_length,
                     self.tree_height: tree_depth,
                     self.tokens_targets: tokens,
                     self.strings_targets: strings,
                     self.W_strings: W_strings,
                     self.B_strings: B_strings}
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
        W_strings = feed_dict[self.W_strings]
        B_strings = feed_dict[self.B_strings]
        dependencies = (strings_targets, W_strings, B_strings)
        targets, dependencies = greedy_correct(targets, outputs, dependencies)
        # targets, dependencies = nearest_correct(targets, outputs, dependencies, outputs, 0.5)
        strings_targets, W_strings, B_strings = dependencies
        feed_dict[self.tokens_targets] = targets
        feed_dict[self.strings_targets] = strings_targets
        feed_dict[self.W_strings] = W_strings
        feed_dict[self.B_strings] = B_strings
        return feed_dict

    def pretrain(self):
        pass

    def train(self):
        try:
            heads = ("epoch", "time", "train_accuracy", "train_loss", "validation_accuracy", "validation_loss")
            formats = ("d", ".4f", ".4f", ".4f", ".4f", ".4f")
            formatter = Formatter(heads, formats, (9, 20, 20, 20, 21, 21), range(6), 10)
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
                    deviation_train_loss = np.sqrt(np.var(trains_loss))
                    validation_loss = np.mean(validations_loss)
                    deviation_validation_loss = np.sqrt(np.var(validations_loss))
                    train_accuracy = np.mean(trains_accuracy)
                    validation_accuracy = np.mean(validations_accuracy)
                    validation_false_negative = np.mean(validations_false_negative)
                    validation_false_positive = np.mean(validations_false_positive)
                    validation_true_negative = np.mean(validations_true_negative)
                    validation_true_positive = np.mean(validations_true_positive)
                    train_loss_graph.append(epoch, train_loss, 3 * 2 * deviation_train_loss)
                    validation_loss_graph.append(epoch, validation_loss, 3 * 2 * deviation_validation_loss)
                    smoothed_train_accuracy_graph.append(epoch, train_accuracy)
                    smoothed_validation_accuracy_graph.append(epoch, validation_accuracy)
                    smoothed_false_negative_graph.append(epoch, validation_false_negative)
                    smoothed_false_positive_graph.append(epoch, validation_false_positive)
                    smoothed_true_negative_graph.append(epoch, validation_true_negative)
                    smoothed_true_positive_graph.append(epoch, validation_true_positive)
                    formatter.print(epoch, delay, train_accuracy, train_loss, validation_accuracy, validation_loss)
                    figure.draw()
                    figure.save(self.folder_path + "/train.png")
                    writer.add_summary(session.run(self.summaries))
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

    def test(self, model_path: str = None):
        heads = ("accuracy", "tokens loss", "strings loss", "target", *(["output", "prob"] * TOP))
        formats = (*([".4f"] * 3), "s", *(["s", ".4f"] * TOP))
        sizes = (*([15] * 3), *([13] * (1 + 2 * TOP)))
        formatter = Formatter(heads, formats, sizes, range(4 + 2 * TOP))
        heads = ("label", "text")
        formats = ("s", "s")
        sizes = (formatter.row_size(0), formatter.size - formatter.row_size(0) - 3)
        appendix_formatter = Formatter(heads, formats, sizes, (0, 1))
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
                fetches = (
                    self.tokens, self.strings, self.tokens_loss, self.strings_loss, self.top_tokens, self.attention)
                tokens, strings, tokens_loss, strings_loss, top_outputs, attention = session.run(fetches, feed_dict)
                inputs = feed_dict[self.inputs]
                num_conditions = feed_dict[self.num_conditions]
                num_tokens = feed_dict[self.num_tokens]
                tokens_targets = feed_dict[self.tokens_targets]
                attention = transpose_attention(attention)
                top_outputs_indices = top_outputs.indices
                top_outputs_probabilities = top_outputs.values
                accuracy, errors = batch_accuracy(tokens_targets, tokens)
                for i in range(BATCH_SIZE):
                    formatter.print_head()
                    for j in range(num_conditions):
                        for k in range(num_tokens):
                            output_target = tokens_targets[i][j][k]
                            _top_outputs_indices = top_outputs_indices[i][j][k]
                            _top_outputs_probabilities = top_outputs_probabilities[i][j][k]
                            output_target = Embeddings.tokens().get_name(output_target)
                            _top_outputs_indices = (Embeddings.tokens().get_name(i) for i in _top_outputs_indices)
                            outputs = (e for p in zip(_top_outputs_indices, _top_outputs_probabilities) for e in p)
                            formatter.print(accuracy[i], tokens_loss[i], strings_loss[i], output_target, *outputs)
                        appendix_formatter.print_delimiter()
                        for token, string in zip(tokens[i][j], strings[i][j]):
                            token = Embeddings.tokens().get_name(token)
                            string = " ".join(Embeddings.words().get_name(word) for word in string)
                            appendix_formatter.print(token, string)
                        appendix_formatter.print_delimiter()
                        print_doc(appendix_formatter, inputs[i], attention[i][j])
                        if j < num_conditions - 1:
                            appendix_formatter.print_delimiter()
                        else:
                            appendix_formatter.print_lower_delimiter()
