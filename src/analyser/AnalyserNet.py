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
from analyser.analyser_rnn import *
from analyser.misc import cross_entropy_loss, l2_loss, greedy_correct, transpose_attention, calc_scores, print_diff, \
    print_scores, newest
from contants import PAD, NOP
from logger import logger
from utils import Score, dumpers
from utils.Formatter import Formatter
from utils.SummaryWriter import SummaryWriter
from utils.wrappers import trace, Timer

flags = tf.app.flags

flags.DEFINE_integer('epochs', None, '')
flags.DEFINE_integer('batch_size', None, '')
flags.DEFINE_integer('input_state_size', None, '')
flags.DEFINE_integer('input_hidden_size', None, '')
flags.DEFINE_integer('token_state_size', None, '')
flags.DEFINE_integer('string_state_size', None, '')
flags.DEFINE_float('l2_weight', None, '')

flags.DEFINE_integer('minimum_length', None, '')
flags.DEFINE_float('train_set', None, '')
flags.DEFINE_float('validation_set', None, '')
flags.DEFINE_float('test_set', None, '')

flags.DEFINE_string('model_dir', None, '')
flags.DEFINE_string('data_set_path', None, '')
flags.DEFINE_string('summaries_dir', None, '')

FLAGS = flags.FLAGS

DataSet = namedtuple("DataSet", ("train", "validation", "test"))


class AnalyserNet:
    @trace("BUILD NET")
    def __init__(self):
        num_words = len(Embeddings.words())
        num_tokens = len(Embeddings.tokens())
        nop = Embeddings.tokens().get_index(NOP)
        pad = Embeddings.words().get_index(PAD)
        with tf.variable_scope("Input"), Timer("BUILD INPUT"):
            self.inputs = tf.placeholder(tf.int32, [FLAGS.batch_size, None], "inputs")
            self.inputs_length = tf.placeholder(tf.int32, [FLAGS.batch_size], "inputs_length")
            self.output_length = tf.placeholder(tf.int32, [], "output_length")
            self.string_length = tf.placeholder(tf.int32, [], "string_length")
            self.token_length = tf.placeholder(tf.int32, [], "token_length")
        with tf.variable_scope("Analyser") as scope, Timer("BUILD BODY"):
            input_cell = (GRUCell(FLAGS.input_state_size), GRUCell(FLAGS.input_state_size))
            tree_output_cell = GRUCell(FLAGS.token_state_size)
            string_output_cell = GRUCell(FLAGS.string_state_size)
            input = tf.gather(tf.constant(np.asarray(Embeddings.words().idx2emb)), self.inputs)
            attention_states = sequence_input(*input_cell, input, self.inputs_length, FLAGS.input_hidden_size)
            self.tokens_logits, self.raw_tokens, states, self.attention = tree_output(
                tree_output_cell, attention_states, self.output_length, self.token_length, num_tokens)
            self.strings_logits, self.raw_strings = string_output(
                string_output_cell, attention_states, states, self.string_length, num_words)
            self.variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope.name)
        with tf.variable_scope("Output"), Timer("BUILD OUTPUT"):
            self.tokens = tf.argmax(self.raw_tokens, 3)
            self.strings = tf.argmax(self.raw_strings, 4)
        with tf.variable_scope("Loss"), Timer("BUILD LOSS"):
            self.tokens_targets = tf.placeholder(tf.int32, [FLAGS.batch_size, None, None], "tokens")
            self.strings_targets = tf.placeholder(tf.int32, [FLAGS.batch_size, None, None, None], "strings")
            self.tokens_loss = cross_entropy_loss(self.tokens_targets, self.tokens_logits, nop)
            self.strings_loss = cross_entropy_loss(self.strings_targets, self.strings_logits, pad)
            self.l2_loss = FLAGS.l2_weight * l2_loss(self.variables)
            self.loss = self.tokens_loss + self.strings_loss + self.l2_loss
        with tf.variable_scope("Optimizer"), Timer("BUILD OPTIMISER"):
            self.optimizer = tf.train.AdamOptimizer().minimize(self.loss)
        with tf.variable_scope("Summaries"), Timer("BUILD SUMMARIES"):
            self.summaries = self.add_variable_summaries()
        self.saver = tf.train.Saver(var_list=self.variables)
        self.save_path = FLAGS.model_dir

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

    @staticmethod
    def get_data_set() -> DataSet:
        data_set = dumpers.pkl_load(FLAGS.data_set_path)
        data_set_length = len(data_set)
        not_allocated = data_set_length
        test_set_length = min(not_allocated, int(data_set_length * FLAGS.test_set))
        not_allocated -= test_set_length
        train_set_length = min(not_allocated, int(data_set_length * FLAGS.train_set))
        not_allocated -= train_set_length
        validation_set_length = min(not_allocated, int(data_set_length * FLAGS.validation_set))
        not_allocated -= validation_set_length
        if test_set_length < FLAGS.minimum_length:
            args = (test_set_length, FLAGS.minimum_length)
            raise ValueError("Length of the test set is very small, length = %d < %d" % args)
        if train_set_length < FLAGS.minimum_length:
            args = (train_set_length, FLAGS.minimum_length)
            raise ValueError("Length of the train set is very small, length = %d < %d" % args)
        if validation_set_length < FLAGS.minimum_length:
            args = (validation_set_length, FLAGS.minimum_length)
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
        inputs, outputs, parameters = batch
        inputs, inputs_length = inputs
        tokens, strings = outputs
        num_conditions, num_tokens, string_length, tree_depth = parameters
        feed_dict = {self.inputs: inputs,
                     self.inputs_length: inputs_length,
                     self.output_length: num_conditions,
                     self.string_length: string_length,
                     self.token_length: tree_depth,
                     self.tokens_targets: tokens,
                     self.strings_targets: strings}
        return feed_dict

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
        device = tf.device('/cpu:0')
        writer = SummaryWriter(FLAGS.summaries_dir, session, self.summaries)
        with session, device, writer, figure:
            session.run(tf.global_variables_initializer())
            best_loss = float("inf")
            for epoch in range(FLAGS.epochs):
                data_set = self.get_data_set()
                start = time.time()
                for batch in data_set.train:
                    feed_dict = self.build_feed_dict(batch)
                    raw_tokens = session.run(self.raw_tokens, feed_dict)
                    feed_dict = self.correct_target(feed_dict, raw_tokens)
                    session.run(self.optimizer, feed_dict)
                trains_loss, train_scores = [], []
                for batch in data_set.train:
                    feed_dict = self.build_feed_dict(batch)
                    raw_tokens = session.run(self.raw_tokens, feed_dict)
                    feed_dict = self.correct_target(feed_dict, raw_tokens)
                    tokens, loss = session.run((self.tokens, self.loss), feed_dict)
                    score = Score.calc(feed_dict[self.tokens_targets], tokens, -1, nop)
                    train_scores.append(score)
                    trains_loss.append(loss)
                validations_loss, validation_scores = [], []
                for batch in data_set.validation:
                    feed_dict = self.build_feed_dict(batch)
                    raw_tokens = session.run(self.raw_tokens, feed_dict)
                    feed_dict = self.correct_target(feed_dict, raw_tokens)
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
        return best_loss

    def test(self) -> (Score, Score, Score, Score):
        config = tf.ConfigProto()
        config.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1
        with tf.Session(config=config) as session, tf.device('/cpu:0'):
            session.run(tf.global_variables_initializer())
            self.restore(session)
            scores = []
            for batch in self.get_data_set().test:
                feed_dict = self.build_feed_dict(batch)
                raw_tokens = session.run(self.raw_tokens, feed_dict)
                feed_dict = self.correct_target(feed_dict, raw_tokens)
                fetches = (self.raw_tokens, self.tokens, self.tokens_loss)
                raw_tokens, tokens, tokens_loss = session.run(fetches, feed_dict)
                fetches = (self.attention, self.strings, self.strings_loss)
                attention, strings, strings_loss = session.run(fetches, feed_dict)
                inputs = feed_dict[self.inputs]
                tokens_targets = feed_dict[self.tokens_targets]
                strings_targets = feed_dict[self.strings_targets]
                attention = transpose_attention(attention)
                scores.append(calc_scores(tokens_targets, tokens, strings_targets, strings))
                print_diff(inputs, attention, raw_tokens, tokens_targets, tokens, strings_targets, strings)
            logger.error(self.save_path)
            scores = print_scores(scores)
        return scores
