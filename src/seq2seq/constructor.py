import random

import numpy as np
import tensorflow as tf
from tensorflow.contrib.legacy_seq2seq.python.ops.seq2seq import attention_decoder
from tensorflow.contrib.rnn.python.ops.core_rnn_cell_impl import GRUCell
from tensorflow.python.ops import variable_scope as vs

from seq2seq.rnn import static_bidirectional_rnn
from utils.wrapper import trace
from utils import dumper
from utils import filter
from variables import *


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


def distance(vector1, vector2):
    # ToDo: return tf.norm(vector1 - vector2)
    return tf.sqrt(tf.reduce_sum(tf.squared_difference(vector1, vector2), 1))


def buildLoss(inputs, outputs, parts: dict):
    errors = [distance(outputs[parts[k]], inp[0]) for k, inp in inputs.items()]
    return tf.reduce_mean(errors)


def buildL2Loss():
    variables = [var for var in tf.global_variables() if var.name in REGULARIZATION_VARIABLES]
    assert len(variables) == len(REGULARIZATION_VARIABLES)
    return tf.reduce_sum([tf.nn.l2_loss(var) for var in variables])


GO = np.zeros([EMB_SIZE], dtype=np.float32)
PAD = np.ones([EMB_SIZE], dtype=np.float32)
INIT = np.zeros([BATCH_SIZE, STATE_SIZE], dtype=np.float32)


@trace
def buildRNN(parts: dict):
    vars_BATCH = {}
    vars_SEQ_SIZES = {}
    attention_states = []
    for label, index in parts.items():
        vars_BATCH[label] = []
        for i in range(MAX_ENCODE_SEQUENCE):
            placeholder = tf.placeholder(tf.float32, [BATCH_SIZE, EMB_SIZE], "batch_{}_{}".format(label, i))
            vars_BATCH[label].append(placeholder)
        vars_SEQ_SIZES[label] = tf.placeholder(tf.int32, [BATCH_SIZE], "sequence_sizes_{}".format(label))
        with vs.variable_scope(label):
            _, output_states_fw, output_states_bw = encoder(vars_BATCH[label], vars_SEQ_SIZES[label], STATE_SIZE)
        attention_states.append(tf.concat(axis=1, values=[output_states_fw[0], output_states_bw[-1]]))
        attention_states.append(tf.concat(axis=1, values=[output_states_fw[-1], output_states_bw[0]]))
    goes = tf.stack([GO for _ in range(BATCH_SIZE)])
    decoder_inputs = [goes] + [goes for _ in range(MAX_DECODE_SEQUENCE - 1)]
    attention_states = tf.stack(attention_states)
    attention_states = tf.transpose(attention_states, [1, 0, 2])
    var_INIT_DECODER_STATE = tf.placeholder(tf.float32, [BATCH_SIZE, STATE_SIZE], "initial_decoder_state")
    res_OUTPUTS, _ = decoder(decoder_inputs, attention_states, STATE_SIZE, var_INIT_DECODER_STATE, EMB_SIZE)
    loss = buildLoss(vars_BATCH, res_OUTPUTS, parts)
    l2_loss = buildL2Loss() * L2_WEIGHT
    train = tf.train.AdamOptimizer(beta1=0.95).minimize(loss + l2_loss)
    return (train, loss, l2_loss), (vars_BATCH, vars_SEQ_SIZES, var_INIT_DECODER_STATE), res_OUTPUTS


@trace
def buildFeedDicts(batches, vars_BATCH, vars_SEQ_SIZES, var_INIT_DECODER_STATE):
    embeddings = list(dumper.load(EMBEDDINGS).values())
    feed_dicts = []
    for _ in range(100):
        feed_dict = {}
        for label in filter.parts.keys():
            vars_BATCH_label = vars_BATCH[label]
            for var_BATCH in vars_BATCH_label:
                feed_dict[var_BATCH] = []
            for _ in range(BATCH_SIZE):
                line = [random.choice(embeddings) for _ in range(MAX_ENCODE_SEQUENCE)]
                for i, word in enumerate(line):
                    feed_dict[vars_BATCH_label[i]].append(word)
            feed_dict[vars_SEQ_SIZES[label]] = tuple(MAX_ENCODE_SEQUENCE for _ in range(BATCH_SIZE))
        feed_dict[var_INIT_DECODER_STATE] = INIT
        feed_dicts.append(feed_dict)
# feed_dicts = []
# for batch in batches:
#     feed_dict = {}
#     for label, (embs, text) in batch.items():
#         vars_BATCH_label = vars_BATCH[label]
#         for var_BATCH in vars_BATCH_label:
#             feed_dict[var_BATCH] = []
#         for emb in embs:
#             line = emb + [PAD for _ in range(MAX_ENCODE_SEQUENCE - len(emb))]
#             for i, word in enumerate(line):
#                 feed_dict[vars_BATCH_label[i]].append(word)
#         feed_dict[vars_SEQ_SIZES[label]] = tuple(len(emb) for emb in embs)
#     feed_dict[var_INIT_DECODER_STATE] = INIT
#     feed_dicts.append(feed_dict)
    return feed_dicts


REGULARIZATION_VARIABLES = (
    "results/bidirectional_rnn/fw/gru_cell/candidate/weights:0",
    "results/bidirectional_rnn/bw/gru_cell/candidate/weights:0",
    "params/bidirectional_rnn/fw/gru_cell/candidate/weights:0",
    "params/bidirectional_rnn/bw/gru_cell/candidate/weights:0",
    "variables/bidirectional_rnn/fw/gru_cell/candidate/weights:0",
    "variables/bidirectional_rnn/bw/gru_cell/candidate/weights:0",
    "head/bidirectional_rnn/fw/gru_cell/candidate/weights:0",
    "head/bidirectional_rnn/bw/gru_cell/candidate/weights:0",
    "attention_decoder/output/weights:0",
    "attention_decoder/gru_cell/candidate/weights:0",
    "attention_decoder/AttnW_0:0",
    "attention_decoder/Attention_0/weights:0",
    "attention_decoder/AttnOutputProjection/weights:0"
)

