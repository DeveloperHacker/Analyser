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


def decoder(inputs, attention_states, state_size: int, initial_state, initial_attention_state=False):
    # ToDo: use this initializers
    # alignment_initializer = tf.truncated_normal_initializer(mean=0.0, stddev=0.001)
    # weighs_initializer = tf.truncated_normal_initializer(mean=0.0, stddev=0.01)
    gru = tf.nn.rnn_cell.GRUCell(state_size)
    return tf.nn.seq2seq.attention_decoder(
        cell=gru,
        dtype=tf.float32,
        decoder_inputs=inputs,
        initial_state=initial_state,
        attention_states=attention_states,
        initial_state_attention=initial_attention_state,
        loop_function=lambda prev, i: prev
    )


def buildLoss(inputs, outputs):
    errs = [tf.sqrt(tf.reduce_sum(tf.squared_difference(outputs[i], inp[0]))) for i, inp in enumerate(inputs)]
    return tf.reduce_mean(errs)


def buildRNN(
        batches: list,
        batch_size: int = BATCH_SIZE,
        state_size: int = STATE_SIZE,
        emb_size: int = FEATURES,
        input_length: int = MAX_ENCODE_SEQUENCE,
        output_length: int = MAX_DECODE_SEQUENCE,
        epoches: int = EPOCHS,
        save_path: str = SEQ2SEQ_MODEL
):
    assert batch_size > 0
    assert state_size > 0
    assert emb_size > 0
    assert input_length > 0
    assert output_length > 0
    assert len(batches) > 0
    parts = batches[0].keys()

    GO = tf.zeros([emb_size], name="GO")
    PAD = tf.ones([emb_size], name="PAD")

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
    attention_states = tf.transpose(attention_states,[1,0,2])
    var_INIT_DECODER_STATE = tf.placeholder(tf.float32, [batch_size, state_size], "initial_decoder_state")
    res_OUTPUTS, res_DECODER_STATE = decoder(decoder_inputs, attention_states, state_size, var_INIT_DECODER_STATE)
    loss = buildLoss(vars_BATCH, res_OUTPUTS)

    feed_dicts = []
    for batch in batches:
        feed_dict = {}
        for label, (embs, text) in batch.items():
            assert len(embs) == batch_size
            assert len(text) == batch_size
            assert all([[len(emb) <= input_length for emb in embs]])
            feed_dict[vars_BATCH[label]] = [emb + [PAD for _ in range(input_length - len(emb))] for emb in embs]
            feed_dict[vars_SEQ_SIZES[label]] = [len(emb) for emb in embs]
        feed_dict[var_INIT_DECODER_STATE] = tf.zeros([batch_size, state_size])
        feed_dicts.append(feed_dict)

    figure = Figure(xauto=True)
    with tf.Session() as session:
        session.run(tf.initialize_all_variables())
        for epoch in range(epoches):
            err = 0
            for feed_dict in feed_dicts:
                err += session.run(loss, feed_dict=feed_dict)
            err /= len(feed_dicts)
            figure.plot(epoch, err)
        saver = tf.train.Saver()
        saver.save(session, save_path)
    figure.close()
