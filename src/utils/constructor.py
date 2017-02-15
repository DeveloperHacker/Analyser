import logging
import sys

import numpy as np
import tensorflow as tf
from tensorflow.python.ops import variable_scope as vs

from Figure import Figure
from mains.variables import *
from utils.rnn import bidirectional_rnn


def encoder(batch, sequence_length, state_size: int, initial_state_fw=None, initial_state_bw=None):
    # ToDo: use this initializer
    # weighs_initializer = tf.truncated_normal_initializer(mean=0.0, stddev=0.01)
    gru_fw = tf.nn.rnn_cell.GRUCell(state_size)
    gru_bw = tf.nn.rnn_cell.GRUCell(state_size)
    return bidirectional_rnn(
        cell_fw=gru_fw,
        cell_bw=gru_bw,
        inputs=batch,
        dtype=tf.float32,
        initial_state_fw=initial_state_fw,
        initial_state_bw=initial_state_bw,
        sequence_length=sequence_length
    )


def decoder(inputs, attention_states, state_size: int, initial_state, emb_size: int, initial_attention_state=False):
    # ToDo: use this initializers
    # alignment_initializer = tf.truncated_normal_initializer(mean=0.0, stddev=0.001)
    # weighs_initializer = tf.truncated_normal_initializer(mean=0.0, stddev=0.01)
    gru = tf.nn.rnn_cell.GRUCell(state_size)
    W_shape = [state_size, emb_size]
    B_shape = [emb_size]
    W = tf.Variable(initial_value=tf.truncated_normal(W_shape, stddev=0.01, dtype=tf.float32), name="W_linear")
    B = tf.Variable(initial_value=tf.truncated_normal(B_shape, stddev=0.01, dtype=tf.float32), name="B_linear")
    decoder_outputs, state = tf.nn.seq2seq.attention_decoder(
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


def buildLoss(inputs, outputs):
    errs = [tf.sqrt(tf.reduce_sum(tf.squared_difference(outputs[i], inp[0]))) for i, inp in enumerate(inputs.values())]
    return tf.reduce_mean(errs)


def buildRNN(
        batches: list,
        batch_size: int = BATCH_SIZE,
        state_size: int = STATE_SIZE,
        emb_size: int = FEATURES,
        input_length: int = MAX_ENCODE_SEQUENCE,
        output_length: int = MAX_DECODE_SEQUENCE,
        epoches: int = SEQ2SEQ_EPOCHS,
        save_path: str = SEQ2SEQ_MODEL,
        log_path: str = SEQ2SEQ_LOG
):
    assert batch_size > 0
    assert state_size > 0
    assert emb_size > 0
    assert input_length > 0
    assert output_length > 0
    assert len(batches) > 0
    parts = batches[0].keys()

    GO = np.zeros([emb_size], dtype=np.float32)
    PAD = np.ones([emb_size], dtype=np.float32)
    INIT = np.zeros([batch_size, state_size], dtype=np.float32)

    vars_BATCH = {}
    vars_SEQ_SIZES = {}
    attention_states = []
    for label in parts:
        vars_BATCH[label] = tuple(
            tf.placeholder(tf.float32, [batch_size, emb_size], "batch_%s" % label) for _ in range(input_length))
        vars_SEQ_SIZES[label] = tf.placeholder(tf.int32, [batch_size], "sequence_sizes_%s" % label)
        with vs.variable_scope(label):
            _, output_states_fw, output_states_bw = encoder(vars_BATCH[label], vars_SEQ_SIZES[label], state_size)
        attention_states.append(tf.concat(1, [output_states_fw[0], output_states_bw[-1]]))
        attention_states.append(tf.concat(1, [output_states_fw[-1], output_states_bw[0]]))
    goes = tf.pack([GO for _ in range(batch_size)])
    decoder_inputs = [goes] + [goes for _ in range(output_length - 1)]
    attention_states = tf.pack(attention_states)
    attention_states = tf.transpose(attention_states, [1, 0, 2])
    var_INIT_DECODER_STATE = tf.placeholder(tf.float32, [batch_size, state_size], "initial_decoder_state")
    res_OUTPUTS, _ = decoder(decoder_inputs, attention_states, state_size, var_INIT_DECODER_STATE, emb_size)
    loss = buildLoss(vars_BATCH, res_OUTPUTS)
    train = tf.train.AdadeltaOptimizer().minimize(loss)

    feed_dicts = []
    for batch in batches:
        feed_dict = {}
        for label, (embs, text) in batch.items():
            assert len(embs) == batch_size
            assert len(text) == batch_size
            assert all([len(emb) <= input_length for emb in embs])
            vars_BATCH_label = vars_BATCH[label]
            for var_BATCH in vars_BATCH_label:
                feed_dict[var_BATCH] = []
            for emb in embs:
                line = emb + [PAD for _ in range(input_length - len(emb))]
                for i, word in enumerate(line):
                    feed_dict[vars_BATCH_label[i]].append(word)
            feed_dict[vars_SEQ_SIZES[label]] = tuple(len(emb) for emb in embs)
        feed_dict[var_INIT_DECODER_STATE] = INIT
        feed_dicts.append(feed_dict)

    figure = Figure(xauto=True)
    logging.basicConfig(level=logging.INFO, filename=log_path)
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    with tf.Session() as session:
        session.run(tf.initialize_all_variables())
        for epoch in range(epoches):
            error = 0
            for feed_dict in feed_dicts:
                local_err, _ = session.run((loss, train), feed_dict=feed_dict)
                error += local_err
            error /= len(feed_dicts)
            figure.plot(epoch, error)
            logging.info("Epoch: %4d Error: %5.4f" % (epoch, error))
        saver = tf.train.Saver()
        saver.save(session, save_path)
    figure.close()
