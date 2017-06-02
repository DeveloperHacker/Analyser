import random
import time

import numpy as np
import tensorflow as tf
from live_plotter.proxy.ProxyFigure import ProxyFigure
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.ops.rnn_cell_impl import GRUCell

from seq2seq.Net import Net
from seq2seq.dynamic_rnn import stack_attention_dynamic_rnn, stack_bidirectional_dynamic_rnn
from utils import dumper
from utils.Formatter import Formatter
from utils.wrapper import trace
from variables.constants import *
from variables.embeddings import WordEmbeddings, TokenEmbeddings
from variables.paths import RESOURCES, ANALYSER, ANALYSER_METHODS
from variables.tags import PARTS

_WEIGHTS_NAME = "weights"
_BIAS_NAME = "biases"


def analyser_rnn(encoder_cells_fw,
                 encoder_cells_bw,
                 word_decoder_cells,
                 token_decoder_cells,
                 inputs: list,
                 sequence_length: list,
                 word_size,
                 token_size,
                 time_steps,
                 num_heads=1,
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

    attention_states = []
    for i, (_inputs, _lengths) in enumerate(zip(inputs, sequence_length)):
        encoder_outputs, states_fw, states_bw = stack_bidirectional_dynamic_rnn(encoder_cells_fw,
                                                                                encoder_cells_bw,
                                                                                _inputs,
                                                                                sequence_length=_lengths,
                                                                                dtype=dtype)
        attention_states.append(tf.concat((states_fw[-1], states_bw[-1]), 2))
    with vs.variable_scope("WordDecoder"):
        decoder_inputs = tf.zeros([time_steps, batch_size, word_size], dtype, "decoder_inputs")
        decoder_outputs, decoder_states = stack_attention_dynamic_rnn(word_decoder_cells,
                                                                      decoder_inputs,
                                                                      attention_states,
                                                                      word_size,
                                                                      num_heads,
                                                                      dtype=dtype)
    with vs.variable_scope("WordSoftmax"):
        W_sft = vs.get_variable(_WEIGHTS_NAME, [word_size, word_size], dtype)
        B_sft = vs.get_variable(_BIAS_NAME, [word_size], dtype, init_ops.constant_initializer(0, dtype))
        decoder_outputs = tf.reshape(decoder_outputs, [time_steps * batch_size, word_size])
        word_logits = decoder_outputs @ W_sft + B_sft
        word_outputs = tf.nn.softmax(word_logits, 1)
        word_outputs = tf.reshape(word_outputs, [time_steps, batch_size, word_size])
    with vs.variable_scope("TokenDecoder"):
        decoder_outputs, decoder_states = stack_attention_dynamic_rnn(token_decoder_cells,
                                                                      decoder_outputs,
                                                                      attention_states,
                                                                      token_size,
                                                                      num_heads,
                                                                      dtype=dtype)
    with vs.variable_scope("TokenSoftmax"):
        W_sft = vs.get_variable(_WEIGHTS_NAME, [token_size, token_size], dtype)
        B_sft = vs.get_variable(_BIAS_NAME, [token_size], dtype, init_ops.constant_initializer(0, dtype))
        decoder_outputs = tf.reshape(decoder_outputs, [time_steps * batch_size, token_size])
        token_logits = decoder_outputs @ W_sft + B_sft
        token_outputs = tf.nn.softmax(token_logits, 1)
        token_outputs = tf.reshape(token_outputs, [time_steps, batch_size, token_size])
    return word_logits, word_outputs, token_logits, token_outputs


class AnalyserNet(Net):
    def __init__(self):
        super().__init__("analyser", ANALYSER)
        self.token_output = None
        self.inputs = None
        self.inputs_sizes = None
        self.token_target = None
        self.word_output = None
        self.word_target = None
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
                    self.inputs.append(tf.placeholder(tf.int32, [BATCH_SIZE, None], "indexes"))
                    self.embeddings.append(tf.gather(embeddings, self.inputs))
                    self.inputs[-1] = tf.stack(self.inputs[-1])
                    self.inputs_sizes.append(tf.placeholder(tf.int32, [BATCH_SIZE], "input_sizes"))
            cells_fw = [GRUCell(ESTATE_SIZE) for _ in range(ENUMBER)]
            cells_bw = [GRUCell(ESTATE_SIZE) for _ in range(ENUMBER)]
            cells_wd = [GRUCell(WDSTATE_SIZE) for _ in range(WDNUMBER)]
            cells_td = [GRUCell(TDSTATE_SIZE) for _ in range(TDNUMBER)]
            self.word_logits, self.word_outputs, self.token_logits, self.token_outputs = analyser_rnn(cells_bw,
                                                                                                      cells_fw,
                                                                                                      cells_wd,
                                                                                                      cells_td,
                                                                                                      self.embeddings,
                                                                                                      self.inputs_sizes,
                                                                                                      WORD_SIZE,
                                                                                                      TOKEN_SIZE)
            self.word_target = tf.placeholder(tf.int32, [BATCH_SIZE, None], "word_target")
            self.token_target = tf.placeholder(tf.int32, [BATCH_SIZE, None], "token_target")
            with vs.variable_scope("losses"):
                logits = self.word_logits
                target = tf.transpose(self.word_target, [1, 0])
                self.word_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=target, logits=logits)
                logits = self.token_logits
                target = tf.transpose(self.token_target, [1, 0])
                self.token_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=target, logits=logits)
                self.loss = tf.reduce_mean(tf.sqrt(tf.square(self.word_loss) + tf.square(self.token_loss)))
            self.optimizer = tf.train.AdamOptimizer().minimize(self.loss)
            self.data_set = dumper.load(ANALYSER_METHODS)
            self.scope = vs.get_variable_scope().name

    def get_data_set(self) -> (list, list):
        random.shuffle(self.data_set)
        train_set = self.data_set[:int(len(self.data_set) * TRAIN_SET)]
        validation_set = self.data_set[-int(len(self.data_set) * VALIDATION_SET):]
        return train_set, validation_set

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
                train_set, validation_set = self.get_data_set()
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
            heads=("loss", "word target", "word output", "token target", "token output"),
            formats=(".4f", "s", "s", "s", "s"),
            sizes=(10, 20, 20, 20, 20),
            rows=(0, 1, 2, 3, 4),
            height=OUTPUT_STEPS
        )
        with tf.Session() as session, tf.device('/cpu:0'):
            self.reset(session)
            self.restore(session)
            train_set, validation_set = self.get_data_set()
            for batch in train_set:
                feed_dict = self.build_feed_dict(batch)
                fetches = (self.loss, self.word_target, self.word_output, self.token_target, self.token_output)
                loss, word_target, word_output, token_target, token_output = session.run(fetches, feed_dict)
                word_target = np.transpose(word_target, [1, 0])
                word_output = np.transpose(word_output, [1, 0, 2])
                token_target = np.transpose(token_target, [1, 0])
                token_output = np.transpose(token_output, [1, 0, 2])
                for wt, wo, tt, to in zip(word_target, word_output, token_target, token_output):
                    for wti, woi, tti, toi in zip(wt, wo, tt, to):
                        word_target = WordEmbeddings.get_word(int(wti))
                        word_output = WordEmbeddings.get_word(int(np.argmax(woi)))
                        token_target = TokenEmbeddings.get_token(int(tti))
                        token_output = TokenEmbeddings.get_token(int(np.argmax(toi)))
                        formatter.print(loss, word_target, word_output, token_target, token_output)

    def build_feed_dict(self, batch) -> dict:
        feed_dict = {}
        inputs, inputs_sizes, word_target, token_target = batch
        for label in PARTS:
            feed_dict.update(zip(self.inputs[label], inputs[label]))
            feed_dict[self.inputs_sizes[label]] = inputs_sizes[label]
        feed_dict.update(zip(self.word_target, word_target))
        feed_dict.update(zip(self.token_target, token_target))
        return feed_dict
