import random
import time

import numpy as np
import tensorflow as tf
from live_plotter.proxy.ProxyFigure import ProxyFigure
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.ops.rnn_cell_impl import GRUCell

from constants.analyser import *
from constants.embeddings import WordEmbeddings, TokenEmbeddings, NUM_TOKENS, NUM_WORDS
from constants.paths import RESOURCES, ANALYSER, ANALYSER_METHODS
from constants.tags import PARTS
from seq2seq.Net import Net
from seq2seq.analyser_rnn import analyser_rnn, input_projection, analysing_loss, sequence_output
from utils import Dumper
from utils.Formatter import Formatter
from utils.wrapper import trace


class AnalyserNet(Net):
    def __init__(self):
        super().__init__("analyser", ANALYSER)
        self.top_token_outputs = None
        self.top_word_outputs = None
        self.decoder_time_steps = None
        self.token_outputs = None
        self.inputs = None
        self.inputs_sizes = None
        self.token_targets = None
        self.word_outputs = None
        self.word_targets = None
        self.loss = None
        self.optimizer = None
        self.data_set = None
        with vs.variable_scope(self.name):
            self.inputs = []
            self.embeddings = []
            self.inputs_sizes = []
            embeddings = tf.constant(np.asarray(WordEmbeddings.idx2emb()))
            for label in PARTS:
                with vs.variable_scope(label):
                    indexes = tf.placeholder(tf.int32, [BATCH_SIZE, None], "indexes")
                    self.embeddings.append(tf.gather(embeddings, indexes))
                    self.inputs.append(indexes)
                    self.inputs_sizes.append(tf.placeholder(tf.int32, [BATCH_SIZE], "input_sizes"))
            self.decoder_time_steps = tf.placeholder(tf.int32, [], "time_steps")
            cells_fw = [GRUCell(ENCODER_STATE_SIZE) for _ in range(NUM_ENCODERS)]
            cells_bw = [GRUCell(ENCODER_STATE_SIZE) for _ in range(NUM_ENCODERS)]
            cells_wd = [GRUCell(WORD_STATE_SIZE) for _ in range(NUM_WORD_DECODERS)]
            cells_td = [GRUCell(TOKEN_STATE_SIZE) for _ in range(NUM_TOKEN_DECODERS)]
            projection = input_projection(self.embeddings, INPUT_SIZE, tf.float32)
            with vs.variable_scope("analyser_rnn"):
                attention_states = analyser_rnn(
                    cells_bw,
                    cells_fw,
                    projection,
                    self.inputs_sizes,
                    tf.float32)
                self.word_logits, self.word_outputs, self.token_logits, self.token_outputs = sequence_output(
                    attention_states,
                    cells_wd,
                    cells_td,
                    NUM_WORDS,
                    NUM_TOKENS,
                    WORD_OUTPUT_SIZE,
                    TOKEN_OUTPUT_SIZE,
                    self.decoder_time_steps,
                    NUM_WORD_HEADS,
                    NUM_TOKEN_HEADS,
                    tf.float32
                )
            self.top_word_outputs = tf.nn.top_k(self.word_outputs, TOP)
            self.top_token_outputs = tf.nn.top_k(self.token_outputs, TOP)
            self.word_targets = tf.placeholder(tf.int32, [BATCH_SIZE, None], "word_target")
            self.token_targets = tf.placeholder(tf.int32, [BATCH_SIZE, None], "token_target")
            self.scope = vs.get_variable_scope().name
            self.loss = analysing_loss(
                self.word_logits,
                self.word_targets,
                self.token_logits,
                self.token_targets,
                self.get_variables())
        self.optimizer = tf.train.AdamOptimizer().minimize(self.loss)
        self.data_set = Dumper.pkl_load(ANALYSER_METHODS)

    def get_data_set(self) -> (list, list, list):
        data_set = list(self.data_set)
        data_set_length = len(self.data_set)
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

    def build_feed_dict(self, batch) -> dict:
        feed_dict = {}
        inputs, inputs_sizes, word_targets, token_targets = batch
        for i, label in enumerate(PARTS):
            feed_dict[self.inputs[i]] = np.asarray(inputs[label]).T
            feed_dict[self.inputs_sizes[i]] = inputs_sizes[label]
        decoder_time_steps = len(word_targets)
        feed_dict[self.word_targets] = word_targets.T
        feed_dict[self.token_targets] = token_targets.T
        feed_dict[self.decoder_time_steps] = decoder_time_steps
        return feed_dict

    @trace
    def pretrain(self):
        pass

    @trace
    def train(self):
        formatter = Formatter(
            heads=("epoch", "time", "train", "validation"),
            formats=("d", ".4f", ".4f", ".4f"),
            sizes=(10, 20, 20, 20),
            rows=(0, 1, 2, 3),
            height=10
        )
        figure = ProxyFigure("train")
        train_loss_graph = figure.fill_graph(1, 1, 1, mode="-ob", color="blue", alpha=0.3)
        validation_loss_graph = figure.fill_graph(1, 1, 1, mode="or", color="red", alpha=0.3)
        figure.set_x_label(1, 1, 1, "epoch")
        figure.set_y_label(1, 1, 1, "loss")

        del tf.get_collection_ref('LAYER_NAME_UIDS')[0]  # suppress dummy warning hack

        with tf.Session() as session, tf.device('/cpu:0'):
            writer = tf.summary.FileWriter(RESOURCES + "/analyser/summary", session.graph)
            self.reset(session)
            for epoch in range(TRAIN_EPOCHS):
                start = time.time()
                train_set, validation_set, test_set = self.get_data_set()
                for batch in train_set:
                    feed_dict = self.build_feed_dict(batch)
                    session.run(self.optimizer, feed_dict)
                train_losses = []
                for batch in train_set:
                    feed_dict = self.build_feed_dict(batch)
                    train_losses.append(session.run(self.loss, feed_dict))
                validation_losses = []
                for batch in validation_set:
                    feed_dict = self.build_feed_dict(batch)
                    validation_losses.append(session.run(self.loss, feed_dict))
                stop = time.time()
                delay = stop - start
                train_loss = np.mean(train_losses)
                deviation_train_loss = np.sqrt(np.var(train_losses))
                validation_loss = np.mean(validation_losses)
                deviation_validation_loss = np.sqrt(np.var(validation_losses))
                formatter.print(epoch, delay, train_loss, validation_loss)
                train_loss_graph.append(epoch, train_loss, deviation_train_loss)
                validation_loss_graph.append(epoch, validation_loss, deviation_validation_loss)
                figure.draw()
                figure.save(self.get_model_path() + "/train.png")
                writer.flush()
                if np.isnan(train_loss) or np.isnan(validation_loss):
                    raise Net.NaNException()
                self.save(session)
        writer.close()

    @trace
    def test(self):
        formatter = Formatter(
            heads=(
                "loss",
                "word target",
                *(["word output"] * TOP),
                *(["prob"] * TOP),
                "token target",
                *(["token output"] * TOP),
                *(["prob"] * TOP)),
            formats=(
                ".4f",
                *(["s"] * (TOP + 1)),
                *([".4f"] * TOP),
                *(["s"] * (TOP + 1)),
                *([".4f"] * TOP)),
            sizes=(
                10,
                *([20] * (TOP + 1)),
                *([10] * TOP),
                *([20] * (TOP + 1)),
                *([10] * TOP)),
            rows=range(3 + 4 * TOP),
            height=30
        )
        with tf.Session() as session, tf.device('/cpu:0'):
            self.reset(session)
            self.restore(session)
            train_set, validation_set, test_set = self.get_data_set()
            for batch in test_set:
                feed_dict = self.build_feed_dict(batch)
                fetches = (
                    self.loss,
                    self.word_targets,
                    self.top_word_outputs,
                    self.token_targets,
                    self.top_token_outputs,
                    self.inputs)
                loss, word_target, top_word_outputs, token_target, top_token_outputs, inputs = session.run(
                    fetches,
                    feed_dict)
                formatter.height = word_target.shape[1]
                top_word_indexes = top_word_outputs.indices
                top_word_probs = top_word_outputs.values
                top_token_indexes = top_token_outputs.indices
                top_token_probs = top_token_outputs.values
                for wt, top_wo_idx, top_wo_prb, tt, top_to_idx, top_to_prb, *inps in zip(
                        word_target,
                        top_word_indexes,
                        top_word_probs,
                        token_target,
                        top_token_indexes,
                        top_token_probs,
                        *inputs):
                    for wti, top_woi_idx, top_woi_prb, tti, top_toi_idx, top_toi_prb in zip(
                            wt,
                            top_wo_idx,
                            top_wo_prb,
                            tt,
                            top_to_idx,
                            top_to_prb):
                        _word_target = WordEmbeddings.get_word(int(wti))
                        _top_word_indexes = [WordEmbeddings.get_word(int(i)) for i in top_woi_idx]
                        _top_word_probs = list(top_woi_prb)
                        _token_target = TokenEmbeddings.get_token(int(tti))
                        _top_token_indexes = [TokenEmbeddings.get_token(int(i)) for i in top_toi_idx]
                        _top_token_probs = list(top_toi_prb)
                        formatter.print(
                            loss,
                            _word_target,
                            *_top_word_indexes,
                            *_top_word_probs,
                            _token_target,
                            *_top_token_indexes,
                            *_top_token_probs)
                    for inp in inps:
                        print(" ".join(WordEmbeddings.get_word(int(i)) for i in inp))
                    print("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^")
