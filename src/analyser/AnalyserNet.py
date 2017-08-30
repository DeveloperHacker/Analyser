import os
import random
import re
import time
from collections import namedtuple

import numpy as np
import tensorflow as tf
from live_plotter.proxy.ProxyFigure import ProxyFigure
from tensorflow.python.ops.rnn_cell_impl import GRUCell

from analyser import Embeddings
from analyser.Options import Options
from analyser.Score import Score
from analyser.analyser_rnn import sequence_input, labels_output, tree_tokens_output, strings_output, \
    sequence_tokens_output
from analyser.attention_dynamic_rnn import Attention
from analyser.misc import cross_entropy_loss, l2_loss, batch_greedy_correct, calc_scores, print_diff, \
    print_scores, newest
from contants import PAD, NOP, UNDEFINED
from logger import logger
from utils import dumpers
from utils.Formatter import Formatter
from utils.SummaryWriter import SummaryWriter
from utils.wrappers import trace, Timer

DataSet = namedtuple("DataSet", ("train", "validation", "test"))


class AnalyserNet:
    @trace("BUILD NET")
    def __init__(self, options: Options, dtype=None, scope=None):
        self.options = options
        self.options.validate()
        num_labels = len(Embeddings.labels())
        num_tokens = len(Embeddings.tokens())
        num_words = len(Embeddings.words())
        undefined = Embeddings.labels().get_index(UNDEFINED)
        nop = Embeddings.tokens().get_index(NOP)
        pad = Embeddings.words().get_index(PAD)
        with tf.variable_scope("Input"), Timer("BUILD INPUT"):
            self.inputs = tf.placeholder(tf.int32, [self.options.batch_size, None], "inputs")
            self.inputs_length = tf.placeholder(tf.int32, [self.options.batch_size], "inputs_length")
            self.labels_length = tf.placeholder(tf.int32, [], "labels_length")
            self.tokens_length = tf.placeholder(tf.int32, [], "tokens_length")
            self.strings_length = tf.placeholder(tf.int32, [], "strings_length")
            inputs = tf.gather(tf.constant(np.asarray(Embeddings.words().idx2emb)), self.inputs)
        with tf.variable_scope(scope or "Analyser", dtype=dtype) as scope, Timer("BUILD BODY"):
            dtype = scope.dtype
            cell_fw = GRUCell(self.options.inputs_state_size)
            cell_bw = GRUCell(self.options.inputs_state_size)
            attention_states = sequence_input(
                cell_fw, cell_bw, inputs, self.inputs_length, self.options.inputs_hidden_size, dtype)
            labels_attention = Attention(
                attention_states, self.options.labels_state_size, dtype=dtype, scope="LabelsAttention")
            labels_cell = GRUCell(self.options.labels_state_size)
            self.labels_logits, self.raw_labels, labels_states, attentions, weights = labels_output(
                labels_cell, labels_attention, num_labels, self.labels_length,
                hidden_size=self.options.labels_hidden_size, dtype=dtype)
            tokens_attention = Attention(
                attention_states, self.options.tokens_state_size, dtype=dtype, scope="TokensAttention")
            if options.tokens_output_type == "tree":
                tokens_left_cell = GRUCell(self.options.tokens_state_size)
                tokens_right_cell = GRUCell(self.options.tokens_state_size)
                tokens_cell = (tokens_left_cell, tokens_right_cell)
                self.tokens_logits, self.raw_tokens, tokens_states, attentions, weights = tree_tokens_output(
                    tokens_cell, tokens_attention, num_tokens, self.tokens_length, labels_states,
                    hidden_size=self.options.tokens_hidden_size, dtype=dtype)
            elif options.tokens_output_type == "sequence":
                tokens_cell = GRUCell(self.options.tokens_state_size)
                self.tokens_logits, self.raw_tokens, tokens_states, attentions, weights = sequence_tokens_output(
                    tokens_cell, tokens_attention, num_tokens, self.tokens_length, labels_states,
                    hidden_size=self.options.tokens_hidden_size, dtype=dtype)
            else:
                raise ValueError("Tokens output type '%s' hasn't recognised" % options.tokens_output_type)
            strings_attention = Attention(
                attention_states, self.options.strings_state_size, dtype=dtype, scope="StringsAttention")
            strings_cell = GRUCell(self.options.strings_state_size)
            self.strings_logits, self.raw_strings, strings_states, attentions, weights = strings_output(
                strings_cell, strings_attention, num_words, self.strings_length, tokens_states,
                hidden_size=self.options.strings_hidden_size, dtype=dtype)
            self.variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope.name)
        with tf.variable_scope("Output"), Timer("BUILD OUTPUT"):
            self.labels = tf.argmax(self.raw_labels, 2)
            self.tokens = tf.argmax(self.raw_tokens, 3)
            self.strings = tf.argmax(self.raw_strings, 4)
        with tf.variable_scope("Loss"), Timer("BUILD LOSS"):
            self.labels_targets = tf.placeholder(tf.int32, [self.options.batch_size, None], "labels")
            self.tokens_targets = tf.placeholder(tf.int32, [self.options.batch_size, None, None], "tokens")
            self.strings_targets = tf.placeholder(tf.int32, [self.options.batch_size, None, None, None], "strings")
            self.labels_loss = cross_entropy_loss(self.labels_targets, self.labels_logits, undefined)
            self.tokens_loss = cross_entropy_loss(self.tokens_targets, self.tokens_logits, nop)
            self.strings_loss = cross_entropy_loss(self.strings_targets, self.strings_logits, pad)
            self.l2_loss = self.options.l2_weight * l2_loss(self.variables)
            self.loss = self.labels_loss + self.tokens_loss + self.strings_loss + self.l2_loss
        with tf.variable_scope("Optimizer"), Timer("BUILD OPTIMISER"):
            self.optimizer = tf.train.AdamOptimizer().minimize(self.loss)
        with tf.variable_scope("Summaries"), Timer("BUILD SUMMARIES"):
            self.summaries = self.add_variable_summaries()
        self.saver = tf.train.Saver(var_list=self.variables)
        self.save_path = self.options.model_dir

    def save(self, session: tf.Session):
        if not os.path.isdir(self.save_path):
            os.mkdir(self.save_path)
        save_time = time.strftime("%d-%m-%Y-%H-%M-%S")
        model_path = os.path.join(self.save_path, "model-%s.ckpt" % save_time)
        self.saver.save(session, model_path)

    def restore(self, session: tf.Session):
        save_path = self.save_path
        model_pattern = re.compile(r"model-\d{1,2}-\d{1,2}-\d{4}-\d{1,2}-\d{1,2}-\d{1,2}\.ckpt\.meta")
        filtrator = lambda path, name: os.path.isfile(path + "/" + name) and re.match(model_pattern, name)
        model_path = newest(save_path, filtrator)
        model_path = ".".join(model_path.split(".")[:-1])
        self.saver.restore(session, model_path)

    def get_data_set(self) -> DataSet:
        data_set = dumpers.pkl_load(self.options.data_set_path)
        data_set_length = len(data_set)
        not_allocated = data_set_length
        test_set_length = min(not_allocated, int(data_set_length * self.options.test_set))
        not_allocated -= test_set_length
        train_set_length = min(not_allocated, int(data_set_length * self.options.train_set))
        not_allocated -= train_set_length
        validation_set_length = min(not_allocated, int(data_set_length * self.options.validation_set))
        not_allocated -= validation_set_length
        if test_set_length < self.options.minimum_length:
            args = (test_set_length, self.options.minimum_length)
            raise ValueError("Length of the test set is very small, length = %d < %d" % args)
        if train_set_length < self.options.minimum_length:
            args = (train_set_length, self.options.minimum_length)
            raise ValueError("Length of the train set is very small, length = %d < %d" % args)
        if validation_set_length < self.options.minimum_length:
            args = (validation_set_length, self.options.minimum_length)
            raise ValueError("Length of the validation set is very small, length = %d < %d" % args)
        test_set = data_set[-test_set_length:]
        data_set = data_set[:-test_set_length]
        random.shuffle(data_set)
        train_set = data_set[-train_set_length:]
        data_set = data_set[:-train_set_length]
        validation_set = data_set[-validation_set_length:]
        return DataSet(train_set, validation_set, test_set)

    def add_variable_summaries(self):
        variables = [tf.reshape(variable, [-1]) for variable in self.variables]
        tf.summary.histogram("Summary", tf.concat(variables, 0))
        for variable in self.variables:
            tf.summary.histogram(variable.name.replace(":", "_"), variable)
        return tf.summary.merge_all()

    def build_feed_dict(self, batch) -> dict:
        (inputs, inputs_length), labels, tokens, strings = batch
        labels_targets, labels_length = labels
        tokens_targets, tokens_length = tokens
        strings_targets, strings_length = strings
        feed_dict = {self.inputs: inputs,
                     self.inputs_length: inputs_length,
                     self.labels_length: labels_length,
                     self.tokens_length: tokens_length,
                     self.strings_length: strings_length,
                     self.labels_targets: labels_targets,
                     self.tokens_targets: tokens_targets,
                     self.strings_targets: strings_targets}
        return feed_dict

    def correct_target(self, feed_dict, session) -> dict:
        fetches = (self.raw_labels, self.raw_tokens, self.raw_strings)
        outputs = session.run(fetches, feed_dict)
        labels_targets = feed_dict[self.labels_targets]
        tokens_targets = feed_dict[self.tokens_targets]
        strings_targets = feed_dict[self.strings_targets]
        undefined = Embeddings.labels().get_index(UNDEFINED)
        nop = Embeddings.tokens().get_index(NOP)
        pad = Embeddings.words().get_index(PAD)
        _labels_targets = np.copy(labels_targets)
        _tokens_targets = np.copy(tokens_targets)
        _strings_targets = np.copy(strings_targets)
        _labels_targets[_labels_targets == -1] = undefined
        _tokens_targets[_tokens_targets == -1] = nop
        _strings_targets[_strings_targets == -1] = pad
        emb_labels_targets = np.asarray(Embeddings.labels().idx2emb)[_labels_targets]
        emb_tokens_targets = np.asarray(Embeddings.tokens().idx2emb)[_tokens_targets]
        num_words = len(Embeddings.words())
        emb_strings_targets = np.eye(num_words)[_strings_targets]
        targets = (emb_labels_targets, emb_tokens_targets, emb_strings_targets)
        dependencies = (labels_targets, tokens_targets, strings_targets)
        targets, dependencies = batch_greedy_correct(targets, outputs, dependencies)
        labels_targets, tokens_targets, strings_targets = dependencies
        feed_dict[self.labels_targets] = labels_targets
        feed_dict[self.tokens_targets] = tokens_targets
        feed_dict[self.strings_targets] = strings_targets
        return feed_dict

    def train(self):
        train_loss_graphs, validation_loss_graphs = [], []
        figure0 = ProxyFigure("loss", self.save_path + "/loss.png")
        loss_labels = ("Labels", "Tokens", "Strings", "Complex")
        for i, label in enumerate(loss_labels):
            train_loss_graphs.append(figure0.smoothed_curve(1, len(loss_labels), i + 1, 0.6, mode="-b"))
            validation_loss_graphs.append(figure0.smoothed_curve(1, len(loss_labels), i + 1, 0.6, mode="-r"))
            figure0.set_x_label(1, len(loss_labels), i + 1, "epoch")
            figure0.set_label(1, len(loss_labels), i + 1, label)
        figure0.set_y_label(1, len(loss_labels), 1, "loss")

        train_score_graphs, validation_score_graphs = [], []
        figure1 = ProxyFigure("score", self.save_path + "/score.png")
        score_labels = ("Labels", "Tokens", "Strings", "Templates", "Code")
        for i, label in enumerate(score_labels):
            train_score_graphs.append(figure1.smoothed_curve(1, len(score_labels), i + 1, 0.6, mode="-b"))
            validation_score_graphs.append(figure1.smoothed_curve(1, len(score_labels), i + 1, 0.6, mode="-r"))
            figure1.set_x_label(1, len(score_labels), i + 1, "epoch")
            figure1.set_label(1, len(score_labels), i + 1, label)
        figure1.set_y_label(1, len(score_labels), 1, "score")

        loss_labels = ("l_" + label for label in loss_labels)
        score_labels = ("s_" + label for label in score_labels)
        labels = [*loss_labels, *score_labels]
        train_labels = ("t_" + label for label in labels)
        validation_labels = ("v_" + label for label in labels)
        heads = ("epoch", "time", *train_labels, *validation_labels)
        formats = ["d", ".4f"] + [".4f"] * 2 * len(labels)
        formatter = Formatter(heads, formats, [15] * len(heads), height=10)

        config = tf.ConfigProto()
        config.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1
        session = tf.Session(config=config)
        # session = tf_debug.LocalCLIDebugWrapperSession(session)
        device = tf.device('/cpu:0')
        writer = SummaryWriter(self.options.summaries_dir, session, self.summaries, session.graph)
        with session, device, writer, figure0, figure1:
            session.run(tf.global_variables_initializer())
            best_loss = float("inf")
            for epoch in range(self.options.epochs):
                data_set = self.get_data_set()
                with Timer(printer=None) as timer:
                    for batch in data_set.train:
                        feed_dict = self.build_feed_dict(batch)
                        feed_dict = self.correct_target(feed_dict, session)
                        session.run(self.optimizer, feed_dict)
                train_losses, train_scores = self.quality(session, data_set.train)
                validation_losses, validation_scores = self.quality(session, data_set.validation)
                train_scores = [score.F_score(1) for score in train_scores]
                validation_scores = [score.F_score(1) for score in validation_scores]
                array = train_losses + train_scores + validation_losses + validation_scores
                formatter.print(epoch, timer.delay(), *array)
                train_loss_is_nan = any(np.isnan(loss) for loss in train_losses)
                validation_loss_is_nan = any(np.isnan(loss) for loss in validation_losses)
                if train_loss_is_nan or validation_loss_is_nan:
                    logger.info("NaN detected")
                    break
                if best_loss > validation_losses[-1] + abs(validation_losses[-1] - train_losses[-1]):
                    best_loss = validation_losses[-1]
                    self.save(session)
                for graph, value in zip(train_loss_graphs, train_losses):
                    graph.append(epoch, value)
                for graph, value in zip(validation_loss_graphs, validation_losses):
                    graph.append(epoch, value)
                for graph, value in zip(train_score_graphs, train_scores):
                    graph.append(epoch, value)
                for graph, value in zip(validation_score_graphs, validation_scores):
                    graph.append(epoch, value)
                figure0.draw()
                figure0.save()
                figure1.draw()
                figure1.save()
                writer.update()

    def test(self):
        config = tf.ConfigProto()
        config.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1
        with tf.Session(config=config) as session, tf.device('/cpu:0'):
            session.run(tf.global_variables_initializer())
            self.restore(session)
            data_set = self.get_data_set()
            losses, scores = self.quality(session, data_set.test)
            for batch in data_set.test:
                feed_dict = self.build_feed_dict(batch)
                feed_dict = self.correct_target(feed_dict, session)
                labels_fetches = (self.labels_targets, self.labels)
                tokens_fetches = (self.tokens_targets, self.tokens)
                strings_fetches = (self.strings_targets, self.strings)
                array = session.run(labels_fetches + tokens_fetches + strings_fetches, feed_dict)
                inputs = feed_dict[self.inputs]
                print_diff(inputs, *array, session.run(self.raw_tokens, feed_dict))
            logger.error(self.save_path)
            print_scores(scores)
        return losses, scores

    def quality(self, session, batches):
        losses, scores = [], []
        for batch in batches:
            feed_dict = self.build_feed_dict(batch)
            feed_dict = self.correct_target(feed_dict, session)
            labels_fetches = (self.labels_targets, self.labels)
            tokens_fetches = (self.tokens_targets, self.tokens)
            strings_fetches = (self.strings_targets, self.strings)
            array = session.run(labels_fetches + tokens_fetches + strings_fetches, feed_dict)
            losses_fetches = (self.labels_loss, self.tokens_loss, self.strings_loss, self.loss)
            losses.append(session.run(losses_fetches, feed_dict))
            scores.append(calc_scores(*array, self.options.flatten_type))
        losses = [np.mean(typed_losses) for typed_losses in zip(*losses)]
        scores = [Score.concat(typed_scores) for typed_scores in zip(*scores)]
        return losses, scores
