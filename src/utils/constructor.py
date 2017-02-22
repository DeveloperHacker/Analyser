import logging

import numpy as np
import tensorflow as tf
from tensorflow.python.ops import variable_scope as vs
from utils.Figure import Figure
from utils.rnn import static_bidirectional_rnn
from utils.wrapper import sigint, SIGINTException, trace
from variables import *
from tensorflow.contrib.rnn.python.ops.core_rnn_cell_impl import GRUCell
from tensorflow.contrib.legacy_seq2seq.python.ops.seq2seq import attention_decoder


@trace
def encoder(batch, sequence_length, state_size: int, initial_state_fw=None, initial_state_bw=None):
    # ToDo: use this initializer
    # weighs_initializer = tf.truncated_normal_initializer(mean=0.0, stddev=0.01)
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
    # ToDo: use this initializers
    # alignment_initializer = tf.truncated_normal_initializer(mean=0.0, stddev=0.001)
    # weighs_initializer = tf.truncated_normal_initializer(mean=0.0, stddev=0.01)
    gru = GRUCell(state_size)
    W_shape = [state_size, emb_size]
    B_shape = [emb_size]
    with vs.variable_scope("Linear"):
        W = tf.Variable(initial_value=tf.truncated_normal(W_shape, stddev=0.01, dtype=tf.float32), name="Matrix")
        B = tf.Variable(initial_value=tf.truncated_normal(B_shape, stddev=0.01, dtype=tf.float32), name="Bias")
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


def tf_distance(vector1, vector2):
    return tf.sqrt(tf.reduce_sum(tf.squared_difference(vector1, vector2), 1))


def buildLoss(inputs, outputs, parts: dict):
    errors = [tf_distance(outputs[parts[k]], inp[0]) for k, inp in inputs.items()]
    return tf.reduce_mean(errors)


def buildL2Loss():
    variables = [var for var in tf.global_variables() if var.name in REGULARIZATION_VARIABLES]
    return tf.reduce_sum([tf.nn.l2_loss(var) for var in variables])


@trace
@sigint
def trainRNN(fetches: tuple, feed_dicts: list, restore: bool = False):
    with tf.Session() as session, Figure(xauto=True) as figure:
        saver = tf.train.Saver()
        try:
            if restore:
                saver.restore(session, SEQ2SEQ_MODEL)
            else:
                session.run(tf.global_variables_initializer())
            for epoch in range(SEQ2SEQ_EPOCHS):
                loss = 0
                l2_loss = 0
                for feed_dict in feed_dicts:
                    _, local_loss, local_l2_loss = session.run(fetches=fetches, feed_dict=feed_dict)
                    loss += local_loss
                    l2_loss += local_l2_loss
                loss /= len(feed_dicts)
                l2_loss /= len(feed_dicts)
                figure.plot(epoch, loss)
                logging.info("Epoch: %4d/%-4d; Loss: %5.4f; L2 loss: %5.4f" % (epoch, SEQ2SEQ_EPOCHS, loss, l2_loss))
                if epoch % 50:
                    saver.save(session, SEQ2SEQ_MODEL)
        except SIGINTException:
            logging.error("SIGINT")
        finally:
            saver.save(session, SEQ2SEQ_MODEL)


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
    feed_dicts = []
    for batch in batches:
        feed_dict = {}
        for label, (embs, text) in batch.items():
            vars_BATCH_label = vars_BATCH[label]
            for var_BATCH in vars_BATCH_label:
                feed_dict[var_BATCH] = []
            for emb in embs:
                line = emb + [PAD for _ in range(MAX_ENCODE_SEQUENCE - len(emb))]
                for i, word in enumerate(line):
                    feed_dict[vars_BATCH_label[i]].append(word)
            feed_dict[vars_SEQ_SIZES[label]] = tuple(len(emb) for emb in embs)
        feed_dict[var_INIT_DECODER_STATE] = INIT
        feed_dicts.append(feed_dict)
    return feed_dicts


REGULARIZATION_VARIABLES = (
    "results/BiRNN/FW/GRUCell/Candidate/Linear/Matrix:0",
    "results/BiRNN/BW/GRUCell/Candidate/Linear/Matrix:0",
    "params/BiRNN/FW/GRUCell/Candidate/Linear/Matrix:0",
    "params/BiRNN/BW/GRUCell/Candidate/Linear/Matrix:0",
    "variables/BiRNN/FW/GRUCell/Candidate/Linear/Matrix:0",
    "variables/BiRNN/BW/GRUCell/Candidate/Linear/Matrix:0",
    "head/BiRNN/FW/GRUCell/Candidate/Linear/Matrix:0",
    "head/BiRNN/BW/GRUCell/Candidate/Linear/Matrix:0",
    "Linear/Matrix:0",
    "attention_decoder/GRUCell/Candidate/Linear/Matrix:0",
    "attention_decoder/AttnW_0:0",
    "attention_decoder/Attention_0/Linear/Matrix:0",
    "attention_decoder/AttnOutputProjection/Linear/Matrix:0"
)


@trace
def initRNN(batches: list, parts: dict):
    fetches, variables, results = buildRNN(parts)
    feed_dicts = buildFeedDicts(batches, *variables)
    return (fetches, variables, results), feed_dicts


def nearest(point, vectors):
    i = np.argmin([np.linalg.norm(point - vector) for vector in vectors])
    return vectors[i]


@trace
def testRNN(vars_BATCH, res_Loss, res_OUTPUTS, feed_dicts, embeddings, parts: dict):
    embeddings = list(embeddings.values())
    vars_FIRST = [None] * 4
    for label, index in parts.items():
        vars_FIRST[index] = vars_BATCH[label][0]
    with tf.Session() as session:
        saver = tf.train.Saver()
        saver.restore(session, SEQ2SEQ_MODEL)
        errors = []
        roundLoss = []
        loss = []
        res_loss = []
        for feed_dict in feed_dicts:
            result = session.run(fetches=[res_OUTPUTS, res_Loss] + vars_FIRST, feed_dict=feed_dict)
            outputs = result[0]
            res_loss.append(result[1])
            targets = result[2:]
            assert len(targets) == 4
            for output, target in zip(outputs[:4], targets):
                for out, tar in zip(output, target):
                    closest = nearest(out, embeddings)
                    errors.append(np.linalg.norm(closest - tar) > 1e-6)
                    roundLoss.append(np.linalg.norm(closest - tar))
                    loss.append(np.linalg.norm(out - tar))
        logging.info("Accuracy: {}%".format((1 - np.mean(errors)) * 100))
        logging.info("RoundLoss: {}".format(np.mean(roundLoss)))
        logging.info("Loss: {}".format(np.mean(loss)))
        logging.info("ResLoss: {}".format(np.mean(res_loss)))
