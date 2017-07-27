import random
import time
from multiprocessing.pool import Pool

import numpy as np
import tensorflow as tf
from live_plotter.proxy.ProxyFigure import ProxyFigure
from tensorflow.python.ops import init_ops
from tensorflow.python.ops.rnn_cell_impl import GRUCell

from constants import embeddings
from constants.analyser import BATCH_SIZE, INPUTS_STATE_SIZE, NUM_ENCODERS, INPUT_SIZE, NUM_WORDS, L2_LOSS_WEIGHT, \
    MINIMUM_DATA_SET_LENGTH, VALIDATION_SET, TRAIN_SET, TEST_SET, SEED, TRAIN_EPOCHS
from constants.paths import SUBSTRACTOR, FULL_DATA_SET, SUBSTRACTOR_GRAPH
from constants.tags import PARTS, PAD
from generate_embeddings import join_java_doc
from prepare_data_set import batching, index_java_doc
from seq2seq import analyser_rnn
from seq2seq.utils import cross_entropy_loss, l2_loss
from seq2seq.Net import Net
from seq2seq.analyser_rnn import sequence_input, input_projection
from seq2seq.dynamic_rnn import attention_dynamic_rnn
from utils import dumpers
from utils import filters
# noinspection PyProtectedMember
from utils.Formatter import Formatter

_WEIGHTS_NAME = analyser_rnn._WEIGHTS_NAME
# noinspection PyProtectedMember
_BIAS_NAME = analyser_rnn._BIAS_NAME

LAYER0_SIZE = 5
LAYER1_SIZE = 10
LAYER2_SIZE = 10
STATE_SIZE = 20


def input_convolution(part_index, from_index, to_index):
    bias_initializer = init_ops.constant_initializer(0, tf.float32)
    with tf.variable_scope("InputConvolutionLayer0Variables"):
        W_layer0 = tf.get_variable(_WEIGHTS_NAME, [1, LAYER0_SIZE])
        B_layer0 = tf.get_variable(_BIAS_NAME, [LAYER0_SIZE], initializer=bias_initializer)
    with tf.variable_scope("InputConvolutionLayer1Variables"):
        W_layer1 = tf.get_variable(_WEIGHTS_NAME, [1, LAYER1_SIZE])
        B_layer1 = tf.get_variable(_BIAS_NAME, [LAYER1_SIZE], initializer=bias_initializer)
    with tf.variable_scope("InputConvolutionLayer2Variables"):
        W_layer2 = tf.get_variable(_WEIGHTS_NAME, [1, LAYER2_SIZE])
        B_layer2 = tf.get_variable(_BIAS_NAME, [LAYER2_SIZE], initializer=bias_initializer)
    with tf.variable_scope("InputConvolutionLayer3Variables"):
        W_layer3 = tf.get_variable(_WEIGHTS_NAME, [LAYER0_SIZE + LAYER1_SIZE + LAYER2_SIZE, STATE_SIZE])
        B_layer3 = tf.get_variable(_BIAS_NAME, [STATE_SIZE], initializer=bias_initializer)
    layer0 = tf.nn.relu_layer(part_index, W_layer0, B_layer0)
    layer1 = tf.nn.relu_layer(from_index, W_layer1, B_layer1)
    layer2 = tf.nn.relu_layer(to_index, W_layer2, B_layer2)
    layer3 = tf.nn.relu_layer(tf.concat((layer0, layer1, layer2), 0), W_layer3, B_layer3)
    return layer3


def build_data_set():
    methods = dumpers.json_load(FULL_DATA_SET)
    with Pool() as pool:
        methods = pool.map(apply, methods)
        methods = [method for method in methods if method is not None]
        batches = pool.map(build_batch, batching(methods))
    random.shuffle(batches)
    return batches


def apply(method):
    try:
        method = filters.apply(method)
        if np.empty(method): return None
        method = join_java_doc(method)
        method = index_java_doc(method)
    except Exception:
        print(method)
        raise ValueError()
    return method


def build_batch(methods):
    inputs_steps = {label: max([len(method["java-doc"][label]) for method in methods]) for label in PARTS}
    docs = {label: [] for label in PARTS}
    docs_sizes = {label: [] for label in PARTS}
    pad = embeddings.words().get_index(PAD)
    part_index = []
    from_index = []
    to_index = []
    outputs = []
    for method in methods:
        part_index.append(random.randrange(0, len(PARTS)))
        for i, label in enumerate(PARTS):
            line = list(method["java-doc"][label])
            if i == part_index[-1]:
                from_index.append(random.randrange(0, len(line)))
                to_index.append(random.randrange(from_index[-1], len(line)))
                outputs.append(line[from_index[-1], to_index[-1]])
            docs_sizes[label].append(len(line))
            expected = inputs_steps[label] + 1 - len(line)
            line = line + [pad] * expected
            docs[label].append(line)
    length = max(len(output) for output in outputs)
    for label in PARTS:
        docs[label] = np.transpose(np.asarray(docs[label]), (1, 0))
        docs_sizes[label] = np.asarray(docs_sizes[label])
    for output in outputs:
        expected = length + 1 - len(output)
        output.extend([pad] * expected)
    outputs = np.asarray(outputs)
    part_index = np.asarray(part_index)
    from_index = np.asarray(from_index)
    to_index = np.asarray(to_index)
    inputs = (docs, docs_sizes, part_index, from_index, to_index)
    return inputs, outputs, length


class SubtractorNet(Net):
    def __init__(self, inputs: tf.Tensor = None):
        super().__init__("substractor", SUBSTRACTOR)
        with tf.variable_scope(self.name):
            _embeddings = tf.constant(np.asarray(embeddings.words().idx2emb))
            self.inputs = {}
            self.inputs_lengths = {}
            doc_embeddings = {}
            cells_fw = {}
            cells_bw = {}
            for label in PARTS:
                label = label[1:]
                indices = tf.placeholder(tf.int32, [BATCH_SIZE, None], "indices_%s" % label)
                self.inputs[label] = indices
                doc_embeddings[label] = tf.gather(_embeddings, indices)
                self.inputs_lengths[label] = tf.placeholder(tf.int32, [BATCH_SIZE], "input_lengths_%s" % label)
                cells_fw[label] = [GRUCell(INPUTS_STATE_SIZE) for _ in range(NUM_ENCODERS)]
                cells_bw[label] = [GRUCell(INPUTS_STATE_SIZE) for _ in range(NUM_ENCODERS)]
            projection = input_projection(doc_embeddings, INPUT_SIZE, tf.float32)
            inputs_states = sequence_input(cells_bw, cells_fw, projection, self.inputs_lengths, tf.float32)
            if inputs is None:
                self.part_index = tf.placeholder(tf.int32, [BATCH_SIZE], "part_index")
                self.from_index = tf.placeholder(tf.int32, [BATCH_SIZE], "from_index")
                self.to_index = tf.placeholder(tf.int32, [BATCH_SIZE], "to_index")
                self.indices = input_convolution(self.part_index, self.from_index, self.to_index)
            else:
                assert len(inputs.get_shape()) == 0
                assert inputs.get_shape()[0] == BATCH_SIZE
                assert inputs.get_shape()[1] == STATE_SIZE
                self.indices = inputs
            strings_cell = GRUCell(STATE_SIZE)
            self.length = tf.placeholder(tf.int32, [], "length")
            self.target = tf.placeholder(tf.int32, [BATCH_SIZE, None], "target")
            decoder_inputs = tf.zeros([self.length, BATCH_SIZE, NUM_WORDS])
            decoder_outputs, decoder_states, attention_weigh = attention_dynamic_rnn(
                strings_cell, decoder_inputs, inputs_states, NUM_WORDS, self.inputs)
            self.logits = tf.transpose(decoder_outputs, [1, 0, 2])
            outputs = tf.reshape(self.logits, [BATCH_SIZE * self.length, NUM_WORDS])
            outputs = tf.nn.softmax(outputs, 1)
            self.outputs = tf.reshape(outputs, [BATCH_SIZE, self.length, NUM_WORDS])
            self.scope = tf.get_variable_scope().name
            self.loss, self.complex_loss = self.build_loss()
        self.optimizer = tf.train.AdamOptimizer().minimize(self.complex_loss)
        self._data_set = build_data_set()

    def build_loss(self):
        with tf.variable_scope("loss"):
            loss = cross_entropy_loss(self.target, self.logits)
            _l2_loss = L2_LOSS_WEIGHT * l2_loss(self.variables)
            complex_loss = loss + _l2_loss
        return loss, complex_loss

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

    def build_feed_dict(self, batch):
        inputs, outputs, length = batch
        (docs, docs_sizes, part_index, from_index, to_index) = inputs
        feed_dict = {}
        for label in PARTS:
            feed_dict[self.inputs[label[1:]]] = np.asarray(docs[label]).T
            feed_dict[self.inputs_lengths[label[1:]]] = docs_sizes[label]
        feed_dict[self.length] = length
        feed_dict[self.target] = outputs
        feed_dict[self.part_index] = part_index
        feed_dict[self.from_index] = from_index
        feed_dict[self.to_index] = to_index
        return feed_dict

    def train(self):
        try:
            heads = ("epoch", "time", "train_loss", "validation_loss")
            formats = ("d", ".4f", ".4f", ".4f")
            formatter = Formatter(heads, formats, (10, 20, 20, 20), range(4), 10)
            figure = ProxyFigure("train")
            validation_loss_graph = figure.distributed_curve(1, 1, 1, mode="-r", color="red", alpha=0.2)
            train_loss_graph = figure.distributed_curve(1, 1, 1, mode="-b", color="blue", alpha=0.2)
            figure.set_y_label(1, 1, 1, "loss")
            figure.set_x_label(1, 1, 3, "epoch")
            figure.set_label(1, 1, 1, "Train and validation losses")
            config = tf.ConfigProto()
            config.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1
            with tf.Session(config=config) as session, tf.device('/cpu:0'):
                session.run(tf.global_variables_initializer())
                for epoch in range(TRAIN_EPOCHS):
                    train_set, validation_set, test_set = self.data_set
                    start = time.time()
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
                    train_loss_graph.append(epoch, train_loss, deviation_train_loss)
                    validation_loss_graph.append(epoch, validation_loss, deviation_validation_loss)
                    formatter.print(epoch, delay, train_loss, validation_loss)
                    figure.draw()
                    figure.save(SUBSTRACTOR_GRAPH)
                    if np.isnan(train_loss) or np.isnan(validation_loss):
                        raise Net.NaNException()
                    self.save(session)
        except Net.NaNException as ex:
            print(ex)
        finally:
            figure.save(SUBSTRACTOR_GRAPH)
            ProxyFigure.destroy()
