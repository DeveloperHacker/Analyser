import tensorflow as tf
from tensorflow.contrib.legacy_seq2seq.python.ops.seq2seq import attention_decoder
from tensorflow.contrib.rnn.python.ops.core_rnn_cell_impl import GRUCell
from tensorflow.python.ops import variable_scope as vs

from seq2seq.rnn import static_bidirectional_rnn
from utils.wrapper import trace
from variables.embeddings import *
from variables.path import SEQ2SEQ
from variables.train import *


@trace
def encoder(batch, sequence_length, state_size: int, initial_state_fw=None, initial_state_bw=None):
    with vs.variable_scope("encoder"):
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
def decoder(
        inputs,
        attention_states,
        state_size: int,
        initial_state,
        output_size: int,
        initial_attention_state: bool = False,
        loop: bool = False
):
    with vs.variable_scope("decoder"):
        gru = GRUCell(state_size)
        W_shape = [state_size, output_size]
        B_shape = [output_size]
        with vs.variable_scope("output"):
            W = tf.Variable(initial_value=tf.truncated_normal(W_shape, stddev=0.01, dtype=tf.float32), name="weights")
            B = tf.Variable(initial_value=tf.truncated_normal(B_shape, stddev=0.01, dtype=tf.float32), name="biases")
        decoder_outputs, state = attention_decoder(
            cell=gru,
            dtype=tf.float32,
            decoder_inputs=inputs,
            initial_state=initial_state,
            attention_states=attention_states,
            initial_state_attention=initial_attention_state,
            loop_function=(lambda prev, i: tf.matmul(prev, W) + B) if loop else None
        )
        outputs = [tf.matmul(decoder_output, W) + B for decoder_output in decoder_outputs]
    return outputs, state


@trace
def analyser():
    with vs.variable_scope("analyser"):
        _INPUTS = {}
        _INPUTS_SIZES = {}
        _INPUTS_STATES = []
        for label in PARTS:
            with vs.variable_scope(label):
                _INPUTS[label] = []
                for i in range(INPUT_SIZE):
                    placeholder = tf.placeholder(tf.float32, [BATCH_SIZE, EMBEDDING_SIZE], "batch_%d" % i)
                    _INPUTS[label].append(placeholder)
                _INPUTS_SIZES[label] = tf.placeholder(tf.int32, [BATCH_SIZE], "input_sizes")
                output_states_fw, output_states_bw = encoder(_INPUTS[label], _INPUTS_SIZES[label], INPUT_STATE_SIZE)
                _INPUTS_STATES.append(tf.concat(axis=1, values=[output_states_fw[0], output_states_bw[-1]]))
                _INPUTS_STATES.append(tf.concat(axis=1, values=[output_states_fw[-1], output_states_bw[0]]))
        goes = tf.stack([GO for _ in range(BATCH_SIZE)])
        decoder_inputs = [goes for _ in range(OUTPUT_SIZE)]
        _INPUTS_STATES = tf.stack(_INPUTS_STATES)
        _INPUTS_STATES = tf.transpose(_INPUTS_STATES, [1, 0, 2])
        _INITIAL_DECODER_STATE = tf.placeholder(tf.float32, [BATCH_SIZE, INPUT_STATE_SIZE], "initial_decoder_state")
        _OUTPUTS, _DECODER_STATES = decoder(decoder_inputs, _INPUTS_STATES, INPUT_STATE_SIZE, _INITIAL_DECODER_STATE,
                                            EMBEDDING_SIZE, loop=True)

    return (_INPUTS, _INPUTS_SIZES, _INITIAL_DECODER_STATE), (_OUTPUTS, _DECODER_STATES)


@trace
def feed_dicts(batches, _BATCH, _SEQ_SIZES, _INIT_DECODER_STATE):
    _feed_dicts = []
    for batch in batches:
        _feed_dict = {}
        for label, (lines, _) in batch.items():
            for _LINE in _BATCH[label]:
                _feed_dict[_LINE] = []
            for embeddings in lines:
                line = embeddings + [PAD for _ in range(INPUT_SIZE - len(embeddings))]
                for i, embedding in enumerate(line):
                    _feed_dict[_BATCH[label][i]].append(embedding)
            _feed_dict[_SEQ_SIZES[label]] = tuple(len(emb) for emb in lines)
        _feed_dict[_INIT_DECODER_STATE] = INITIAL_STATE
        _feed_dicts.append(_feed_dict)
        return _feed_dicts
