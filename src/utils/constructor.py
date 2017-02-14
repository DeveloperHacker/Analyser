import tensorflow as tf

from utils.batcher import vector


def encoder(state_size, batch, sequence_length, initial_state_fw=None, initial_state_bw=None):
    weighs_initializer = tf.truncated_normal_initializer(mean=0.0, stddev=0.01)
    gru_fw = tf.nn.rnn_cell.GRUCell(state_size)
    gru_bw = tf.nn.rnn_cell.GRUCell(state_size)
    return tf.nn.bidirectional_rnn(
        cell_fw=gru_fw,
        cell_bw=gru_bw,
        inputs=batch,
        dtype=tf.float32,
        initial_state_fw=initial_state_fw,
        initial_state_bw=initial_state_bw,
        sequence_length=sequence_length
    )


def decoder(state_size, inputs, initial_state, attention_states, initial_attention_state=False):
    alignment_initializer = tf.truncated_normal_initializer(mean=0.0, stddev=0.001)
    weighs_initializer = tf.truncated_normal_initializer(mean=0.0, stddev=0.01)
    gru = tf.nn.rnn_cell.GRUCell(state_size)
    loop_function = lambda prev, i: prev
    return tf.nn.seq2seq.attention_decoder(
        cell=gru,
        dtype=tf.float32,
        decoder_inputs=inputs,
        initial_state=initial_state,
        attention_states=attention_states,
        initial_state_attention=initial_attention_state,
        loop_function=loop_function
    )


def constructRNNNet(batches: list, state_size: int):
    for batch in batches:
        hidden = {}
        for label in batch.keys():
            data = batch[label]
            _, hidden[label] = encoder(state_size, data, vector(data))
        outputs, states = decoder(state_size, )
