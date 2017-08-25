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
from analyser.analyser_rnn import sequence_input, labels_output, tree_tokens_output, strings_output
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
        with tf.variable_scope(scope or "Analyser", dtype=dtype) as scope, Timer("BUILD BODY"):
            dtype = scope.dtype
            cell_fw = GRUCell(self.options.inputs_state_size)
            cell_bw = GRUCell(self.options.inputs_state_size)
            labels_cell = GRUCell(self.options.labels_state_size)
            tokens_left_cell = GRUCell(self.options.tokens_state_size)
            tokens_right_cell = GRUCell(self.options.tokens_state_size)
            tokens_cell = (tokens_left_cell, tokens_right_cell)
            strings_cell = GRUCell(self.options.strings_state_size)
            inputs = tf.gather(tf.constant(np.asarray(Embeddings.words().idx2emb)), self.inputs)
            attention_states = sequence_input(
                cell_fw, cell_bw, inputs, self.inputs_length, self.options.inputs_hidden_size, dtype)
            labels_attention = Attention(
                attention_states, self.options.labels_state_size, dtype=dtype, scope="LabelsAttention")
            self.labels_logits, self.raw_labels, labels_states, attentions, weights = labels_output(
                labels_cell, labels_attention, num_labels, self.labels_length,
                hidden_size=self.options.labels_hidden_size, dtype=dtype)
            tokens_attention = Attention(
                attention_states, self.options.tokens_state_size, dtype=dtype, scope="TokensAttention")
            self.tokens_logits, self.raw_tokens, tokens_states, attentions, weights = tree_tokens_output(
                tokens_cell, tokens_attention, num_tokens, self.tokens_length, labels_states,
                hidden_size=self.options.tokens_hidden_size, dtype=dtype)
            strings_attention = Attention(
                attention_states, self.options.strings_state_size, dtype=dtype, scope="StringsAttention")
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
        nop = Embeddings.tokens().get_index(NOP)
        heads = ("epoch", "time", "F1", "loss", "F1", "loss")
        formats = ("d", ".4f", ".4f", ".4f", ".4f", ".4f")
        formatter = Formatter(heads, formats, (9, 20, 20, 20, 21, 21), height=10)
        figure = ProxyFigure("train", self.save_path + "/train.png")
        validation_loss_graph = figure.curve(2, 1, 1, mode="-r")
        train_loss_graph = figure.curve(2, 1, 1, mode="-b")
        smoothed_train_f1_graph = figure.smoothed_curve(2, 1, 2, 0.64, mode="-b")
        smoothed_validation_f1_graph = figure.smoothed_curve(2, 1, 2, 0.64, mode="-r")
        figure.set_y_label(2, 1, 1, "loss")
        figure.set_y_label(2, 1, 2, "accuracy")
        figure.set_x_label(2, 1, 2, "epoch")
        figure.set_label(2, 1, 1, "Train and validation losses")
        figure.set_label(2, 1, 2, "Train and validation F1 score")
        config = tf.ConfigProto()
        config.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1
        session = tf.Session(config=config)
        # session = tf_debug.LocalCLIDebugWrapperSession(session)
        device = tf.device('/cpu:0')
        writer = SummaryWriter(self.options.summaries_dir, session, self.summaries, session.graph)
        with session, device, writer, figure:
            session.run(tf.global_variables_initializer())
            best_loss = float("inf")
            for epoch in range(self.options.epochs):
                data_set = self.get_data_set()
                start = time.time()
                for batch in data_set.train:
                    feed_dict = self.build_feed_dict(batch)
                    feed_dict = self.correct_target(feed_dict, session)
                    session.run(self.optimizer, feed_dict)
                trains_loss, train_scores = [], []
                for batch in data_set.train:
                    feed_dict = self.build_feed_dict(batch)
                    feed_dict = self.correct_target(feed_dict, session)
                    tokens, loss = session.run((self.tokens, self.loss), feed_dict)
                    score = Score.calc(feed_dict[self.tokens_targets], tokens, -1, nop)
                    train_scores.append(score)
                    trains_loss.append(loss)
                validations_loss, validation_scores = [], []
                for batch in data_set.validation:
                    feed_dict = self.build_feed_dict(batch)
                    feed_dict = self.correct_target(feed_dict, session)
                    tokens, loss = session.run((self.tokens, self.loss), feed_dict)
                    score = Score.calc(feed_dict[self.tokens_targets], tokens, -1, nop)
                    validation_scores.append(score)
                    validations_loss.append(loss)
                stop = time.time()
                delay = stop - start
                train_loss = np.mean(trains_loss)
                validation_loss = np.mean(validations_loss)
                train_loss_graph.append(epoch, train_loss)
                validation_loss_graph.append(epoch, validation_loss)
                train_score = np.mean([score.F_score(1) for score in train_scores])
                validation_score = np.mean([score.F_score(1) for score in validation_scores])
                smoothed_train_f1_graph.append(epoch, train_score)
                smoothed_validation_f1_graph.append(epoch, validation_score)
                formatter.print(epoch, delay, train_score, train_loss, validation_score, validation_loss)
                if np.isnan(train_loss) or np.isnan(validation_loss):
                    logger.info("NaN detected")
                    break
                if best_loss > validation_loss:
                    best_loss = validation_loss
                    self.save(session)
                figure.draw()
                figure.save()
                writer.update()

    def test(self):
        config = tf.ConfigProto()
        config.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1
        with tf.Session(config=config) as session, tf.device('/cpu:0'):
            session.run(tf.global_variables_initializer())
            self.restore(session)
            losses, scores = [], []
            for batch in self.get_data_set().test:
                feed_dict = self.build_feed_dict(batch)
                feed_dict = self.correct_target(feed_dict, session)
                labels_fetches = (self.labels_targets, self.labels)
                tokens_fetches = (self.tokens_targets, self.tokens)
                strings_fetches = (self.strings_targets, self.strings)
                array = session.run(labels_fetches + tokens_fetches + strings_fetches, feed_dict)
                losses_fetches = (self.labels_loss, self.tokens_loss, self.strings_loss, self.loss)
                losses.append(session.run(losses_fetches, feed_dict))
                scores.append(calc_scores(*array))
                print_diff(*array, session.run(self.raw_tokens, feed_dict))
            logger.error(self.save_path)
            losses = [np.mean(typed_losses) for typed_losses in zip(*losses)]
            scores = [Score.concat(typed_scores) for typed_scores in zip(*scores)]
            print_scores(scores)
        return losses, scores
