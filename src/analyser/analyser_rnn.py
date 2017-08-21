from tensorflow.python.framework import dtypes
from tensorflow.python.ops import array_ops, nn_impl
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import tensor_array_ops as ta_ops
from tensorflow.python.ops import variable_scope as vs

from analyser import dynamic_rnn
from analyser.dynamic_rnn import attention_dynamic_rnn
from analyser.dynamic_rnn import bidirectional_dynamic_rnn

# noinspection PyProtectedMember
_WEIGHTS_NAME = dynamic_rnn._WEIGHTS_NAME
# noinspection PyProtectedMember
_BIAS_NAME = dynamic_rnn._BIAS_NAME


def sequence_input(cell_fw, cell_bw, inputs, inputs_length, hidden_size):
    with vs.variable_scope("SequenceInput"):
        batch_size = inputs.get_shape()[0].value
        assert batch_size == inputs_length.get_shape()[0], "Batch sizes of inputs and inputs lengths must be equals"
        input_length = array_ops.shape(inputs)[1]
        input_size = inputs.get_shape()[2].value

        bias_initializer = init_ops.constant_initializer(0)
        with vs.variable_scope("InputProjectionVariables"):
            W = vs.get_variable(_WEIGHTS_NAME, [input_size, hidden_size])
            B = vs.get_variable(_BIAS_NAME, [hidden_size], initializer=bias_initializer)
        inputs = array_ops.reshape(inputs, [batch_size * input_length, input_size])
        hiddens = nn_impl.relu_layer(inputs, W, B)
        hiddens = array_ops.reshape(hiddens, [batch_size, input_length, hidden_size])

        _, states = bidirectional_dynamic_rnn(cell_fw, cell_bw, hiddens, inputs_length, dtype=dtypes.float32)
        states = array_ops.concat(states, 2)
    return states


# def sequence_output(inputs_states,
#                     root_cells, root_time_steps, root_num_heads,
#                     sequence_cells, sequence_length, sequence_num_heads,
#                     num_labels, num_tokens,
#                     dtype):
#     if len(sequence_cells) != len(root_cells):
#         raise ValueError("Number of sequence cells and root cells must be equals")
#     batch_size = None
#     for states in inputs_states:
#         _batch_size = states.get_shape()[0].value
#         if batch_size is not None and _batch_size != batch_size:
#             raise ValueError("Batch sizes for any Attention states must be equals")
#         batch_size = _batch_size
#     with vs.variable_scope("root_decoder"):
#         root_decoder_inputs = tf.zeros([root_time_steps, batch_size, num_tokens], dtype)
#         root_decoder_outputs, root_decoder_states, attention_mask = stack_attention_dynamic_rnn(
#             root_cells, root_decoder_inputs, inputs_states, num_labels, root_num_heads, dtype=dtype)
#         roots = root_decoder_states[-1]
#         labels = root_decoder_outputs
#         num_conditions = roots.get_shape()[0].value
#         if num_conditions is None:
#             num_conditions = array_ops.shape(roots)[0]
#         root_decoder_states = tf.transpose(tf.stack(root_decoder_states), [1, 0, 2, 3])
#     with vs.variable_scope("sequence_decoder"):
#         time = array_ops.constant(0, tf.int32, name="time")
#         with vs.variable_scope("Arrays"):
#             outputs_ta = tf.TensorArray(dtype, num_conditions, tensor_array_name="outputs", clear_after_read=False)
#             states_ta = tf.TensorArray(dtype, num_conditions, tensor_array_name="states", clear_after_read=False)
#
#         def time_step(time, outputs_ta, states_ta):
#             sequence_decoder_inputs = tf.zeros([sequence_length, batch_size, num_tokens], dtype)
#             initial_states = tf.unstack(tf.gather(root_decoder_states, time))
#             sequence_decoder_outputs, sequence_decoder_states, sequence_decoder_attention_weighs = stack_attention_dynamic_rnn(
#                 sequence_cells,
#                 sequence_decoder_inputs,
#                 inputs_states,
#                 num_tokens,
#                 sequence_num_heads,
#                 initial_states=initial_states,
#                 dtype=dtype)
#             outputs_ta = outputs_ta.write(time, sequence_decoder_outputs[-1])
#             states_ta = states_ta.write(time, sequence_decoder_states[-1])
#             return time + 1, outputs_ta, states_ta
#
#         _, outputs_ta, states_ta = control_flow_ops.while_loop(
#             cond=lambda time, *_: time < num_conditions, body=time_step, loop_vars=(time, outputs_ta, states_ta))
#
#         with vs.variable_scope("LabelProjection"):
#             labels_logits = tf.reshape(labels, [num_conditions * batch_size, num_labels])
#             labels = tf.nn.softmax(labels_logits, 1)
#             labels_logits = tf.reshape(labels_logits, [num_conditions, batch_size, num_labels])
#             labels = tf.reshape(labels, [num_conditions, batch_size, num_labels])
#         with vs.variable_scope("OutputProjection"):
#             states = states_ta.stack()
#             outputs_logits = outputs_ta.stack()
#             outputs_logits = tf.reshape(outputs_logits, [num_conditions * sequence_length * batch_size, num_tokens])
#             outputs = tf.nn.softmax(outputs_logits, 1)
#             outputs_logits = tf.reshape(outputs_logits, [num_conditions, sequence_length, batch_size, num_tokens])
#             outputs = tf.reshape(outputs, [num_conditions, sequence_length, batch_size, num_tokens])
#         labels_logits = tf.transpose(labels_logits, [1, 0, 2])
#         labels = tf.transpose(labels, [1, 0, 2])
#         outputs_logits = tf.transpose(outputs_logits, [2, 0, 1, 3])
#         outputs = tf.transpose(outputs, [2, 0, 1, 3])
#         states = tf.transpose(states, [2, 0, 1, 3])
#     return (labels_logits, labels, outputs_logits, outputs), states, attention_mask


def tree_output(cell, attention_states, num_trees, tree_height, output_size):
    with vs.variable_scope("TreeOutput"):
        batch_size = attention_states.get_shape()[0].value
        state_size = cell.state_size
        hidden_size = cell.output_size
        output_length = 2 ** tree_height - 1

        bias_initializer = init_ops.constant_initializer(0)
        with vs.variable_scope("OutputProjectionVariables"):
            W_out = vs.get_variable(_WEIGHTS_NAME, [state_size, output_size])
            B_out = vs.get_variable(_BIAS_NAME, [output_size], initializer=bias_initializer)
        with vs.variable_scope("LeftStateProjectionVariables"):
            W_left = vs.get_variable(_WEIGHTS_NAME, [state_size, state_size])
            B_left = vs.get_variable(_BIAS_NAME, [state_size], initializer=bias_initializer)
        with vs.variable_scope("RightStateProjectionVariables"):
            W_right = vs.get_variable(_WEIGHTS_NAME, [state_size, state_size])
            B_right = vs.get_variable(_BIAS_NAME, [state_size], initializer=bias_initializer)
        with vs.variable_scope("Arrays"):
            states_ta = ta_ops.TensorArray(
                dtypes.float32, output_length, tensor_array_name="states", clear_after_read=False)

        decoder_inputs = array_ops.zeros([num_trees, batch_size, hidden_size])
        decoder_outputs, states, attention = attention_dynamic_rnn(
            cell, decoder_inputs, attention_states, hidden_size, loop_function=lambda output, _: output)

        def time_step(time, size, states_ta):
            with vs.variable_scope("Time-Step"):
                state = states_ta.read(time)
                state = array_ops.reshape(state, [num_trees * batch_size, state_size])
                left_state = state @ W_left + B_left
                right_state = state @ W_right + B_right
                left_state = array_ops.reshape(left_state, [num_trees, batch_size, state_size])
                right_state = array_ops.reshape(right_state, [num_trees, batch_size, state_size])
                states_ta = states_ta.write(size, left_state)
                states_ta = states_ta.write(size + 1, right_state)
            return time + 1, size + 2, states_ta

        time_steps = 2 ** (tree_height - 1) - 1
        states_ta = states_ta.write(0, states)
        _, _, states_ta = control_flow_ops.while_loop(lambda time, *_: time < time_steps, time_step, (0, 1, states_ta))

        states = states_ta.stack()
        logits = array_ops.reshape(states, [output_length * num_trees * batch_size, state_size])
        logits = logits @ W_out + B_out
        outputs = nn_ops.softmax(logits, 1)
        logits = array_ops.reshape(logits, [output_length, num_trees, batch_size, output_size])
        outputs = array_ops.reshape(outputs, [output_length, num_trees, batch_size, output_size])
        logits = array_ops.transpose(logits, [2, 1, 0, 3])
        outputs = array_ops.transpose(outputs, [2, 1, 0, 3])
        states = array_ops.transpose(states, [2, 1, 0, 3])
    return logits, outputs, states, attention


def string_output(cell, attention_states, states, output_length, output_size):
    with vs.variable_scope("StringsOutput"):
        batch_size = states.get_shape()[0].value
        output_height = array_ops.shape(states)[1]
        output_width = array_ops.shape(states)[2]
        state_size = states.get_shape()[3].value
        num_outputs = output_height * output_width
        states = array_ops.reshape(states, [batch_size, num_outputs, state_size])
        states = array_ops.transpose(states, [1, 0, 2])
        hidden_size = cell.output_size

        bias_initializer = init_ops.constant_initializer(0)
        with vs.variable_scope("HiddenLayerVariables"):
            W_hidden = vs.get_variable(_WEIGHTS_NAME, [state_size, cell.state_size])
            B_hidden = vs.get_variable(_BIAS_NAME, [cell.state_size], initializer=bias_initializer)
        with vs.variable_scope("OutputLayerVariables"):
            W_out = vs.get_variable(_WEIGHTS_NAME, [hidden_size, output_size])
            B_out = vs.get_variable(_BIAS_NAME, [output_size], initializer=bias_initializer)
        with vs.variable_scope("Arrays"):
            hiddens_ta = ta_ops.TensorArray(dtypes.float32, num_outputs, tensor_array_name="outputs")

        def time_step(time, hiddens_ta):
            with vs.variable_scope("Time-Step"):
                decoder_inputs = array_ops.zeros([output_length, batch_size, hidden_size])
                initial_state = nn_impl.relu_layer(array_ops.gather(states, time), W_hidden, B_hidden)
                hidden, _, attention = attention_dynamic_rnn(
                    cell, decoder_inputs, attention_states, hidden_size, initial_state,
                    loop_function=lambda output, _: output)
                hiddens_ta = hiddens_ta.write(time, hidden)
            return time + 1, hiddens_ta

        _, hiddens_ta = control_flow_ops.while_loop(lambda time, *_: time < num_outputs, time_step, (0, hiddens_ta))

        hiddens = hiddens_ta.stack()
        logits = array_ops.reshape(hiddens, [num_outputs * output_length * batch_size, hidden_size])
        logits = logits @ W_out + B_out
        outputs = nn_ops.softmax(logits, 1)
        logits = array_ops.reshape(logits, [output_height, output_width, output_length, batch_size, output_size])
        outputs = array_ops.reshape(outputs, [output_height, output_width, output_length, batch_size, output_size])
        logits = array_ops.transpose(logits, [3, 0, 1, 2, 4])
        outputs = array_ops.transpose(outputs, [3, 0, 1, 2, 4])
    return logits, outputs
