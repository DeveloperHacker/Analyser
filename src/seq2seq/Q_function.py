import tensorflow as tf
from tensorflow.python.ops import variable_scope as vs

from seq2seq.seq2seq import build_encoder, build_decoder
from utils.wrapper import trace
from variables.sintax import *
from variables.train import *


@trace
def build_q_function(inputs, inputs_sizes, initial_decoder_state, output):
    with vs.variable_scope("q-function"):
        inputs_states = []
        for label in PARTS:
            with vs.variable_scope(label):
                states_fw, states_bw = build_encoder(inputs[label], INPUT_STATE_SIZE, inputs_sizes[label])
                inputs_states.append(tf.concat(axis=1, values=[states_fw[0], states_bw[-1]]))
                inputs_states.append(tf.concat(axis=1, values=[states_fw[-1], states_bw[0]]))
        with vs.variable_scope("output"):
            output = tf.transpose(output, [1, 0, 2])
            states_fw, states_bw = build_encoder(tf.unstack(output), OUTPUT_STATE_SIZE)
            output_states = [tf.concat(axis=1, values=[states_fw[i], states_bw[-(i + 1)]]) for i in range(OUTPUT_SIZE)]
        inputs_states = tf.transpose(tf.stack(inputs_states), [1, 0, 2])
        Q, _ = build_decoder(output_states, inputs_states, INPUT_STATE_SIZE, initial_decoder_state, 1)
        Q = tf.transpose(tf.reshape(tf.stack(Q), [OUTPUT_SIZE, BATCH_SIZE]), [1, 0])
        with tf.variable_scope("softmax"):
            W_shape = [OUTPUT_STATE_SIZE * 2, 1]
            B_shape = [1]
            W = tf.Variable(initial_value=tf.truncated_normal(W_shape, stddev=0.01, dtype=tf.float32), name="weights")
            B = tf.Variable(initial_value=tf.truncated_normal(B_shape, stddev=0.01, dtype=tf.float32), name="biases")
            output_states = tf.reshape(tf.stack(output_states), [BATCH_SIZE * OUTPUT_SIZE, OUTPUT_STATE_SIZE * 2])
            I = tf.nn.softmax(tf.reshape(tf.matmul(output_states, W) + B, [BATCH_SIZE, OUTPUT_SIZE]))
            q = tf.reduce_sum(Q * I, axis=1)
    return q


@trace
def placeholders():
    inputs = {}
    inputs_sizes = {}
    for label in PARTS:
        with vs.variable_scope(label):
            inputs[label] = []
            for i in range(INPUT_SIZE):
                placeholder = tf.placeholder(tf.float32, [BATCH_SIZE, EMBEDDING_SIZE], "batch_%d" % i)
                inputs[label].append(placeholder)
                inputs_sizes[label] = tf.placeholder(tf.int32, [BATCH_SIZE], "input_sizes")
    initial_decoder_state = tf.placeholder(tf.float32, [BATCH_SIZE, INPUT_STATE_SIZE], "initial_decoder_state")
    with vs.variable_scope("output"):
        output = []
        for i in range(OUTPUT_SIZE):
            placeholder = tf.placeholder(tf.float32, [BATCH_SIZE, EMBEDDING_SIZE], "output_%d" % i)
            output.append(placeholder)
        output_sizes = tf.placeholder(tf.int32, [BATCH_SIZE], "output_sizes")
    return inputs, inputs_sizes, output, output_sizes, initial_decoder_state
