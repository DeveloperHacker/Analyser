import tensorflow as tf
from tensorflow.python.ops import variable_scope as vs

from seq2seq.analyser import encoder, decoder
from utils.wrapper import trace
from variables.path import SEQ2SEQ
from variables.sintax import *
from variables.train import *


@trace
def Q_function():
    with vs.variable_scope("Q-function"):
        _INPUTS = {}
        _INPUTS_SIZES = {}
        _INPUTS_STATES = []
        _OUTPUT_STATES = []
        for label in PARTS:
            with vs.variable_scope(label):
                _INPUTS[label] = []
                for i in range(INPUT_SIZE):
                    placeholder = tf.placeholder(tf.float32, [BATCH_SIZE, EMBEDDING_SIZE], "input_%d" % i)
                    _INPUTS[label].append(placeholder)
                _INPUTS_SIZES[label] = tf.placeholder(tf.int32, [BATCH_SIZE], "input_sizes")
                _STATES_FW, _STATES_BW = encoder(_INPUTS[label], _INPUTS_SIZES[label], INPUT_STATE_SIZE)
                _INPUTS_STATES.append(tf.concat(axis=1, values=[_STATES_FW[0], _STATES_BW[-1]]))
                _INPUTS_STATES.append(tf.concat(axis=1, values=[_STATES_FW[-1], _STATES_BW[0]]))
        with vs.variable_scope("output"):
            _OUTPUT = []
            for i in range(OUTPUT_SIZE):
                placeholder = tf.placeholder(tf.float32, [BATCH_SIZE, EMBEDDING_SIZE], "output_%d" % i)
                _OUTPUT.append(placeholder)
            _OUTPUT_SIZES = tf.placeholder(tf.int32, [BATCH_SIZE], "output_sizes")
            _STATES_FW, _STATES_BW = encoder(_OUTPUT, _OUTPUT_SIZES, OUTPUT_STATE_SIZE)
            for i in range(OUTPUT_SIZE):
                _OUTPUT_STATES.append(tf.concat(axis=1, values=[_STATES_FW[i], _STATES_BW[-(i + 1)]]))
        _INPUTS_STATES = tf.transpose(tf.stack(_INPUTS_STATES), [1, 0, 2])
        _INITIAL_DECODER_STATE = tf.placeholder(tf.float32, [BATCH_SIZE, INPUT_STATE_SIZE], "initial_decoder_state")
        _Q, _DECODER_STATES = decoder(_OUTPUT_STATES, _INPUTS_STATES, INPUT_STATE_SIZE, _INITIAL_DECODER_STATE, 1)
        _Q = tf.transpose(tf.reshape(tf.stack(_Q), [OUTPUT_SIZE, BATCH_SIZE]), [1, 0])
        with tf.variable_scope("softmax"):
            W_shape = [OUTPUT_STATE_SIZE * 2, 1]
            B_shape = [1]
            W = tf.Variable(initial_value=tf.truncated_normal(W_shape, stddev=0.01, dtype=tf.float32), name="weights")
            B = tf.Variable(initial_value=tf.truncated_normal(B_shape, stddev=0.01, dtype=tf.float32), name="biases")
            _OUTPUT_STATES = tf.reshape(tf.stack(_OUTPUT_STATES), [BATCH_SIZE * OUTPUT_SIZE, OUTPUT_STATE_SIZE * 2])
            _I = tf.nn.softmax(tf.reshape(tf.matmul(_OUTPUT_STATES, W) + B, [BATCH_SIZE, OUTPUT_SIZE]))
        _RESULT = tf.reduce_sum(_Q * _I, axis=1)

        exit(1)
        return _RESULT
