import numpy as np
import tensorflow as tf
from tensorflow.contrib.legacy_seq2seq.python.ops.seq2seq import attention_decoder
from tensorflow.contrib.rnn.python.ops.core_rnn_cell_impl import GRUCell
from tensorflow.python.ops import variable_scope as vs

from seq2seq.rnn import static_bidirectional_rnn
from utils.wrapper import trace
from variables.embeddings import TOKENS
from variables.train import *


@trace
def encoder(batch, sequence_length, state_size: int, initial_state_fw=None, initial_state_bw=None):
    gru_fw = GRUCell(state_size)
    gru_bw = GRUCell(state_size)
    return static_bidirectional_rnn(
        cell_fw=gru_fw,
        cell_bw=gru_bw,
        inputs=batch,
        dtype=tf.float32,
        initial_state_fw=initial_state_fw,
        initial_state_bw=initial_state_bw,
        sequence_length=sequence_length
    )


@trace
def decoder(inputs, attention_states, state_size: int, initial_state, emb_size: int, initial_attention_state=False):
    gru = GRUCell(state_size)
    W_shape = [state_size, emb_size]
    B_shape = [emb_size]
    with vs.variable_scope("attention_decoder/output"):
        W = tf.Variable(initial_value=tf.truncated_normal(W_shape, stddev=0.01, dtype=tf.float32), name="weights")
        B = tf.Variable(initial_value=tf.truncated_normal(B_shape, stddev=0.01, dtype=tf.float32), name="biases")
    decoder_outputs, state = attention_decoder(
        cell=gru,
        dtype=tf.float32,
        decoder_inputs=inputs,
        initial_state=initial_state,
        attention_states=attention_states,
        initial_state_attention=initial_attention_state,
        loop_function=lambda prev, i: linear(prev, W, B)
    )
    outputs = [linear(decoder_output, W, B) for decoder_output in decoder_outputs]
    return outputs, state


def linear(inp, W, B):
    return tf.matmul(inp, W) + B


GO = np.zeros([EMB_SIZE], dtype=np.float32)
PAD = np.ones([EMB_SIZE], dtype=np.float32)
INIT = np.zeros([BATCH_SIZE, STATE_SIZE], dtype=np.float32)


@trace
def build_rnn(parts: dict):
    _INPUTS = {}
    _SEQ_SIZES = {}
    attention_states = []
    for label, _ in parts.items():
        with vs.variable_scope(label):
            _INPUTS[label] = []
            for i in range(INPUT_SIZE):
                placeholder = tf.placeholder(tf.float32, [BATCH_SIZE, EMB_SIZE], "batch_%d" % i)
                _INPUTS[label].append(placeholder)
            _SEQ_SIZES[label] = tf.placeholder(tf.int32, [BATCH_SIZE], "sequence_sizes")
            _, output_states_fw, output_states_bw = encoder(_INPUTS[label], _SEQ_SIZES[label], STATE_SIZE)
            attention_states.append(tf.concat(axis=1, values=[output_states_fw[0], output_states_bw[-1]]))
            attention_states.append(tf.concat(axis=1, values=[output_states_fw[-1], output_states_bw[0]]))
    goes = tf.stack([GO for _ in range(BATCH_SIZE)])
    decoder_inputs = [goes] + [goes for _ in range(OUTPUT_SIZE - 1)]
    attention_states = tf.stack(attention_states)
    attention_states = tf.transpose(attention_states, [1, 0, 2])
    _INITIAL_DECODER_STATE = tf.placeholder(tf.float32, [BATCH_SIZE, STATE_SIZE], "initial_decoder_state")
    _OUTPUTS, _DECODER_STATES = decoder(decoder_inputs, attention_states, STATE_SIZE, _INITIAL_DECODER_STATE, EMB_SIZE)
    return (_INPUTS, _SEQ_SIZES, _INITIAL_DECODER_STATE), (_OUTPUTS, _DECODER_STATES)


def l2_loss():
    _VARIABLES = [var for var in tf.global_variables() if var.name in REGULARIZATION_VARIABLES]
    assert len(_VARIABLES) == len(REGULARIZATION_VARIABLES)
    return tf.reduce_sum([tf.nn.l2_loss(var) for var in _VARIABLES])


def distance(vector1, vector2):
    # ToDo: return tf.norm(vector1 - vector2)
    return tf.sqrt(tf.reduce_sum(tf.squared_difference(vector1, vector2), 1))


@trace
def fetches(_INPUTS, _OUTPUTS):
    embeddings = tf.stack(tuple(embedding for _, embedding in TOKENS))
    contains = lambda x: [tf.reduce_min(distance(vector, embeddings)) for vector in x]
    _CONTAINS_LOSS = tf.reduce_mean(tf.map_fn(contains, _OUTPUTS))
    _VARIANCE_LOSS = 1.0 / tf.reduce_mean(tf.square(tf.nn.moments(tf.stack(_OUTPUTS), [0])[1]))
    _L2_LOSS = l2_loss()
    _LOSS = tf.sqrt(
        tf.square(CONTAINS_WEIGHT * _CONTAINS_LOSS) +
        tf.square(VARIANCE_WEIGHT * _VARIANCE_LOSS) +
        tf.square(L2_WEIGHT * _L2_LOSS))
    _TRAIN = tf.train.AdamOptimizer(beta1=0.90).minimize(_LOSS)
    return _TRAIN, _LOSS, _CONTAINS_LOSS, _VARIANCE_LOSS, _L2_LOSS


@trace
def build_feed_dicts(batches, _BATCH, _SEQ_SIZES, _INIT_DECODER_STATE):
    _feed_dicts = []
    for batch in batches:
        _feed_dict = {}
        for label, (embs, text) in batch.items():
            vars_BATCH_label = _BATCH[label]
            for var_BATCH in vars_BATCH_label:
                _feed_dict[var_BATCH] = []
            for emb in embs:
                line = emb + [PAD for _ in range(INPUT_SIZE - len(emb))]
                for i, word in enumerate(line):
                    _feed_dict[vars_BATCH_label[i]].append(word)
            _feed_dict[_SEQ_SIZES[label]] = tuple(len(emb) for emb in embs)
        _feed_dict[_INIT_DECODER_STATE] = INIT
        _feed_dicts.append(_feed_dict)
        return _feed_dicts
