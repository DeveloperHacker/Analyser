import tensorflow as tf
from tensorflow.contrib.legacy_seq2seq import attention_decoder
from tensorflow.contrib.rnn import GRUCell
from tensorflow.python.ops import variable_scope as vs

from seq2seq.rnn import static_bidirectional_rnn
from utils.wrapper import trace
from variables.train import INITIALIZATION_STD


@trace
def build_encoder(batch, state_size: int, sequence_length=None, initial_state_fw=None, initial_state_bw=None):
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
def build_decoder(
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
        W = tf.Variable(initial_value=tf.truncated_normal(W_shape, stddev=INITIALIZATION_STD, dtype=tf.float32),
                        name="weights")
        B = tf.Variable(initial_value=tf.truncated_normal(B_shape, stddev=INITIALIZATION_STD, dtype=tf.float32),
                        name="biases")
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
def build_l2_loss(trainable_variables, regularisation_variable_names):
    with vs.variable_scope("l2_loss"):
        variables = [variable for variable in trainable_variables if variable.name in regularisation_variable_names]
        assert len(variables) == len(regularisation_variable_names)
        l2_loss = tf.reduce_sum([tf.nn.l2_loss(variable) for variable in variables])
    return l2_loss
