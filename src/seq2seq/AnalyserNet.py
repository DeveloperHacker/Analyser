import random
import time
import warnings

import numpy as np
import tensorflow as tf
from live_plotter.proxy.ProxyFigure import ProxyFigure
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.ops.rnn_cell_impl import GRUCell

from constants.analyser import *
from constants.embeddings import WordEmbeddings, TokenEmbeddings, NUM_TOKENS, NUM_WORDS
from constants.paths import RESOURCES, ANALYSER, ANALYSER_METHODS
from constants.tags import PARTS
from seq2seq.Net import Net
from seq2seq.dynamic_rnn import stack_attention_dynamic_rnn, stack_bidirectional_dynamic_rnn
from utils import dumper
from utils.Formatter import Formatter
from utils.wrapper import trace

_WEIGHTS_NAME = "weights"
_BIAS_NAME = "biases"


def analyser_rnn(encoder_cells_fw,
                 encoder_cells_bw,
                 word_decoder_cells,
                 token_decoder_cells,
                 inputs: list,
                 sequence_length: list,
                 num_words,
                 num_tokens,
                 word_size,
                 token_size,
                 time_steps,
                 word_num_heads,
                 token_num_heads,
                 dtype=None):
    if len(inputs) != len(sequence_length):
        raise ValueError("Number of inputs and inputs lengths must be equals")
    if len(inputs) == 0:
        raise ValueError("Number of inputs must be greater zero")
    batch_size = None
    for _inputs, _sequence_length in zip(inputs, sequence_length):
        size = _inputs.get_shape()[0].value
        if batch_size is not None and size != batch_size:
            raise ValueError("Batch sizes of Inputs and inputs lengths must be equals")
        batch_size = size

    if not dtype:
        dtype = tf.float32

    with vs.variable_scope("analyser_rnn"):
        attention_states = []
        for i, (_inputs, _lengths) in enumerate(zip(inputs, sequence_length)):
            encoder_outputs, states_fw, states_bw = stack_bidirectional_dynamic_rnn(
                encoder_cells_fw,
                encoder_cells_bw,
                _inputs,
                sequence_length=_lengths,
                dtype=dtype)
            attention_states.append(tf.concat((states_fw[-1], states_bw[-1]), 2))
        with vs.variable_scope("WordDecoder"):
            decoder_inputs = tf.zeros([time_steps, batch_size, word_size], dtype, "decoder_inputs")
            decoder_outputs, decoder_states = stack_attention_dynamic_rnn(
                word_decoder_cells,
                decoder_inputs,
                attention_states,
                word_size,
                word_num_heads,
                dtype=dtype)
        with vs.variable_scope("WordSoftmax"):
            W_sft = vs.get_variable(_WEIGHTS_NAME, [word_size, num_words], dtype)
            B_sft = vs.get_variable(_BIAS_NAME, [num_words], dtype, init_ops.constant_initializer(0, dtype))
            word_logits = tf.reshape(decoder_outputs, [time_steps * batch_size, word_size])
            word_logits = word_logits @ W_sft + B_sft
            word_outputs = tf.nn.softmax(word_logits, 1)
            word_logits = tf.reshape(word_logits, [time_steps, batch_size, num_words])
            word_outputs = tf.reshape(word_outputs, [time_steps, batch_size, num_words])
            word_logits = tf.transpose(word_logits, [1, 0, 2])
            word_outputs = tf.transpose(word_outputs, [1, 0, 2])
        with vs.variable_scope("TokenDecoder"):
            decoder_outputs, decoder_states = stack_attention_dynamic_rnn(
                token_decoder_cells,
                decoder_outputs,
                attention_states,
                token_size,
                token_num_heads,
                use_inputs=True,
                dtype=dtype)
        with vs.variable_scope("TokenSoftmax"):
            W_sft = vs.get_variable(_WEIGHTS_NAME, [token_size, num_tokens], dtype)
            B_sft = vs.get_variable(_BIAS_NAME, [num_tokens], dtype, init_ops.constant_initializer(0, dtype))
            token_logits = tf.reshape(decoder_outputs, [time_steps * batch_size, token_size])
            token_logits = token_logits @ W_sft + B_sft
            token_outputs = tf.nn.softmax(token_logits, 1)
            token_outputs = tf.reshape(token_outputs, [time_steps, batch_size, num_tokens])
            token_logits = tf.reshape(token_logits, [time_steps, batch_size, num_tokens])
            token_outputs = tf.reshape(token_outputs, [time_steps, batch_size, num_tokens])
            token_logits = tf.transpose(token_logits, [1, 0, 2])
            token_outputs = tf.transpose(token_outputs, [1, 0, 2])
    return word_logits, word_outputs, token_logits, token_outputs


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
            self.word_logits, self.word_outputs, self.token_logits, self.token_outputs = analyser_rnn(
                cells_bw,
                cells_fw,
                cells_wd,
                cells_td,
                self.embeddings,
                self.inputs_sizes,
                NUM_WORDS,
                NUM_TOKENS,
                WORD_OUTPUT_SIZE,
                TOKEN_OUTPUT_SIZE,
                self.decoder_time_steps,
                NUM_WORD_HEADS,
                NUM_TOKEN_HEADS)
            self.top_word_outputs = tf.nn.top_k(self.word_outputs, TOP)
            self.top_token_outputs = tf.nn.top_k(self.token_outputs, TOP)
            self.word_targets = tf.placeholder(tf.int32, [BATCH_SIZE, None], "word_target")
            self.token_targets = tf.placeholder(tf.int32, [BATCH_SIZE, None], "token_target")
            with vs.variable_scope("losses"):
                logits = self.word_logits
                targets = self.word_targets
                self.word_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=targets, logits=logits)
                logits = self.token_logits
                targets = self.token_targets
                self.token_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=targets, logits=logits)
                self.loss = tf.reduce_mean(tf.sqrt(tf.square(self.word_loss) + tf.square(self.token_loss)))
            self.optimizer = tf.train.AdamOptimizer().minimize(self.loss)
            self.data_set = dumper.load(ANALYSER_METHODS)
            self.scope = vs.get_variable_scope().name

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
            warnings.warn("\n\r\tWarning: Length of the test set is very small, length = %d < %d" % args)
        if train_set_length < MINIMUM_DATA_SET_LENGTH:
            args = (train_set_length, MINIMUM_DATA_SET_LENGTH)
            warnings.warn("\n\r\tWarning: Length of the train set is very small, length = %d < %d" % args)
        if validation_set_length < MINIMUM_DATA_SET_LENGTH:
            args = (validation_set_length, MINIMUM_DATA_SET_LENGTH)
            warnings.warn("\n\r\tWarning: Length of the validation set is very small, length = %d < %d" % args)
        test_set = data_set[-test_set_length:]
        data_set = data_set[:-test_set_length]
        random.shuffle(data_set)
        train_set = data_set[-train_set_length:]
        data_set = data_set[:-train_set_length]
        validation_set = data_set[-validation_set_length:]
        return train_set, validation_set, test_set

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
        validation_loss_graph = figure.fill_graph(1, 1, 1, mode="-or", color="red", alpha=0.3)
        figure.set_x_label(1, 1, 1, "epoch")
        figure.set_y_label(1, 1, 1, "loss")
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
                var_train_loss = np.var(train_losses)
                validation_loss = np.mean(validation_losses)
                var_validation_loss = np.var(validation_losses)
                formatter.print(epoch, delay, train_loss, validation_loss)
                train_loss_graph.append(epoch, train_loss, var_train_loss)
                validation_loss_graph.append(epoch, validation_loss, var_validation_loss)
                figure.draw()
                self.save(session)
                figure.save(self.get_model_path() + "/train.png")
            writer.flush()
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
                    self.top_token_outputs)
                loss, word_target, top_word_outputs, token_target, top_token_outputs = session.run(fetches, feed_dict)
                formatter.set_height(word_target.shape[1])
                top_word_indexes = top_word_outputs.indices
                top_word_probs = top_word_outputs.values
                top_token_indexes = top_token_outputs.indices
                top_token_probs = top_token_outputs.values
                for wt, top_wo_idx, top_wo_prb, tt, top_to_idx, top_to_prb in zip(
                        word_target,
                        top_word_indexes,
                        top_word_probs,
                        token_target,
                        top_token_indexes,
                        top_token_probs):
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
