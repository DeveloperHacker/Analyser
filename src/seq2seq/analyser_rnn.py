import tensorflow as tf
from tensorflow.python.ops import control_flow_ops, array_ops, variable_scope as vs, init_ops

from seq2seq import dynamic_rnn
from seq2seq.dynamic_rnn import stack_attention_dynamic_rnn, stack_bidirectional_dynamic_rnn, attention_dynamic_rnn

# noinspection PyProtectedMember
_WEIGHTS_NAME = dynamic_rnn._WEIGHTS_NAME
# noinspection PyProtectedMember
_BIAS_NAME = dynamic_rnn._BIAS_NAME


def sequence_input(cells_fw: dict, cells_bw: dict, inputs: dict, sequence_lengths: dict, dtype):
    if len(inputs) != len(sequence_lengths):
        raise ValueError("Number of inputs and inputs lengths must be equals")
    if len(inputs) == 0:
        raise ValueError("Number of inputs must be greater zero")

    batch_size = None
    for label, _inputs in inputs.items():
        _sequence_lengths = sequence_lengths[label]
        _batch_size = _inputs.get_shape()[0].value
        if batch_size is not None and _batch_size != batch_size:
            raise ValueError("Batch sizes for any inputs must be equals")
        batch_size = _sequence_lengths.get_shape()[0].value
        if _batch_size != batch_size:
            raise ValueError("Batch sizes of Inputs and inputs lengths must be equals")

    states = []
    for label, _inputs in inputs.items():
        _sequence_lengths = sequence_lengths[label]
        _cells_fw = cells_fw[label]
        _cells_bw = cells_bw[label]
        with vs.variable_scope("encoder_%s" % label):
            encoder_outputs, states_fw, states_bw = stack_bidirectional_dynamic_rnn(
                _cells_fw, _cells_bw, _inputs, sequence_length=_sequence_lengths, dtype=dtype)
            states.append(tf.concat((states_fw[-1], states_bw[-1]), 2))
    return states


def input_projection(inputs, projection_size, dtype):
    input_size = None
    for _inputs in inputs.values():
        _input_size = _inputs.get_shape()[2].value
        if input_size is not None and _input_size != input_size:
            raise ValueError("Input sizes for any inputs must be equals")
        input_size = _input_size

    with vs.variable_scope("input_projection"):
        W_enc = vs.get_variable(_WEIGHTS_NAME, [input_size, projection_size], dtype)
        B_enc = vs.get_variable(_BIAS_NAME, [projection_size], dtype, init_ops.constant_initializer(0, dtype))
        projections = {}
        for label, _inputs in inputs.items():
            batch_size = _inputs.get_shape()[0].value
            input_length = _inputs.get_shape()[1].value
            if input_length is None:
                input_length = tf.shape(_inputs)[1]
            _inputs = tf.reshape(_inputs, [batch_size * input_length, input_size])
            projection = tf.nn.relu(_inputs @ W_enc + B_enc)
            projection = tf.reshape(projection, [batch_size, input_length, projection_size])
            projections[label] = projection
    return projections


def sequence_output(inputs_states,
                    root_cells, root_time_steps, root_num_heads,
                    sequence_cells, sequence_time_steps, sequence_num_heads,
                    num_labels, num_tokens,
                    dtype):
    if len(sequence_cells) != len(root_cells):
        raise ValueError("Number of sequence cells and root cells must be equals")
    batch_size = None
    for states in inputs_states:
        _batch_size = states.get_shape()[0].value
        if batch_size is not None and _batch_size != batch_size:
            raise ValueError("Batch sizes for any Attention states must be equals")
        batch_size = _batch_size
    with vs.variable_scope("root_decoder"):
        root_decoder_inputs = tf.zeros([root_time_steps, batch_size, num_tokens], dtype)
        root_decoder_outputs, root_decoder_states, attention_mask = stack_attention_dynamic_rnn(
            root_cells, root_decoder_inputs, inputs_states, num_labels, root_num_heads, dtype=dtype)
        roots = root_decoder_states[-1]
        labels = root_decoder_outputs
        num_roots = roots.get_shape()[0].value
        if num_roots is None:
            num_roots = array_ops.shape(roots)[0]
        root_decoder_states = tf.transpose(tf.stack(root_decoder_states), [1, 0, 2, 3])
    with vs.variable_scope("sequence_decoder"):
        time_steps = sequence_time_steps + 1
        time = array_ops.constant(0, tf.int32, name="time")
        with vs.variable_scope("Arrays"):
            outputs_ta = tf.TensorArray(dtype, num_roots, tensor_array_name="outputs", clear_after_read=False)
            states_ta = tf.TensorArray(dtype, num_roots, tensor_array_name="states", clear_after_read=False)

        def time_step(time, outputs_ta, states_ta):
            sequence_decoder_inputs = tf.zeros([time_steps, batch_size, num_tokens], dtype)
            initial_states = tf.unstack(tf.gather(root_decoder_states, time))
            sequence_decoder_outputs, sequence_decoder_states, sequence_decoder_attention_weighs = stack_attention_dynamic_rnn(
                sequence_cells,
                sequence_decoder_inputs,
                inputs_states,
                num_tokens,
                sequence_num_heads,
                initial_states=initial_states,
                dtype=dtype)
            outputs_ta = outputs_ta.write(time, sequence_decoder_outputs[-1])
            states_ta = states_ta.write(time, sequence_decoder_states[-1])
            return time + 1, outputs_ta, states_ta

        _, outputs_ta, states_ta = control_flow_ops.while_loop(
            cond=lambda time, *_: time < num_roots, body=time_step, loop_vars=(time, outputs_ta, states_ta))

        with vs.variable_scope("LabelProjection"):
            labels_logits = tf.reshape(labels, [num_roots * batch_size, num_labels])
            labels = tf.nn.softmax(labels_logits, 1)
            labels_logits = tf.reshape(labels_logits, [num_roots, batch_size, num_labels])
            labels = tf.reshape(labels, [num_roots, batch_size, num_labels])
        with vs.variable_scope("OutputProjection"):
            states = states_ta.stack()
            outputs_logits = outputs_ta.stack()
            outputs_logits = tf.reshape(outputs_logits, [time_steps * num_roots * batch_size, num_tokens])
            outputs = tf.nn.softmax(outputs_logits, 1)
            outputs_logits = tf.reshape(outputs_logits, [time_steps, num_roots, batch_size, num_tokens])
            outputs = tf.reshape(outputs, [time_steps, num_roots, batch_size, num_tokens])
        labels_logits = tf.transpose(labels_logits, [1, 0, 2])
        labels = tf.transpose(labels, [1, 0, 2])
        outputs_logits = tf.transpose(outputs_logits, [2, 1, 0, 3])
        outputs = tf.transpose(outputs, [2, 1, 0, 3])
        states = tf.transpose(states, [2, 1, 0, 3])
    return (labels_logits, labels, outputs_logits, outputs), states, attention_mask


def tree_output(attention_states,
                conditions_cells, num_conditions, num_conditions_attention_heads,
                tree_height,
                num_labels, num_tokens,
                dtype):
    batch_size = None
    for state in attention_states:
        _batch_size = state.get_shape()[0].value
        if batch_size is not None and _batch_size != batch_size:
            raise ValueError("Batch sizes for any Attention states must be equals")
        batch_size = _batch_size

    with vs.variable_scope("condition_decoder"):
        decoder_inputs = tf.zeros([num_conditions, batch_size, num_tokens], dtype)
        decoder_outputs, decoder_states, attention_mask = stack_attention_dynamic_rnn(
            conditions_cells,
            decoder_inputs,
            attention_states,
            num_labels,
            num_conditions_attention_heads,
            dtype=dtype)
        conditions = decoder_states[-1]
        labels = decoder_outputs
        num_conditions = array_ops.shape(conditions)[0]

    with vs.variable_scope("tree_decoder"):
        state_size = conditions.get_shape()[2].value
        sequence_length = 2 ** tree_height - 1
        bias_initializer = init_ops.constant_initializer(0, dtype)
        with vs.variable_scope("OutputProjectionVariables"):
            W_out = vs.get_variable(_WEIGHTS_NAME, [state_size, num_tokens], dtype)
            B_out = vs.get_variable(_BIAS_NAME, [num_tokens], dtype, bias_initializer)
        with vs.variable_scope("LeftStateProjectionVariables"):
            W_left = vs.get_variable(_WEIGHTS_NAME, [state_size, state_size], dtype)
            B_left = vs.get_variable(_BIAS_NAME, [state_size], dtype, bias_initializer)
        with vs.variable_scope("RightStateProjectionVariables"):
            W_right = vs.get_variable(_WEIGHTS_NAME, [state_size, state_size], dtype)
            B_right = vs.get_variable(_BIAS_NAME, [state_size], dtype, bias_initializer)
        with vs.variable_scope("Arrays"):
            states_ta = tf.TensorArray(dtype, sequence_length, tensor_array_name="states", clear_after_read=False)

        def time_step(time, size, states_ta):
            state = states_ta.read(time)
            state = tf.reshape(state, [num_conditions * batch_size, state_size])
            left_state = state @ W_left + B_left
            right_state = state @ W_right + B_right
            left_state = tf.reshape(left_state, [num_conditions, batch_size, state_size])
            right_state = tf.reshape(right_state, [num_conditions, batch_size, state_size])
            states_ta = states_ta.write(size, left_state)
            states_ta = states_ta.write(size + 1, right_state)
            return time + 1, size + 2, states_ta

        time_steps = 2 ** (tree_height - 1) - 1
        states_ta = states_ta.write(0, conditions)
        _, _, states_ta = control_flow_ops.while_loop(lambda time, *_: time < time_steps, time_step, (0, 1, states_ta))
        with vs.variable_scope("LabelProjection"):
            labels_logits = tf.reshape(labels, [num_conditions * batch_size, num_labels])
            labels = tf.nn.softmax(labels_logits, 1)
            labels_logits = tf.reshape(labels_logits, [num_conditions, batch_size, num_labels])
            labels = tf.reshape(labels, [num_conditions, batch_size, num_labels])
        with vs.variable_scope("OutputProjection"):
            states = states_ta.stack()
            tokens_logits = tf.reshape(states, [sequence_length * num_conditions * batch_size, state_size])
            tokens_logits = tokens_logits @ W_out + B_out
            tokens = tf.nn.softmax(tokens_logits, 1)
            tokens_logits = tf.reshape(tokens_logits, [sequence_length, num_conditions, batch_size, num_tokens])
            tokens = tf.reshape(tokens, [sequence_length, num_conditions, batch_size, num_tokens])
        states = tf.transpose(states, [2, 1, 0, 3])
        labels_logits = tf.transpose(labels_logits, [1, 0, 2])
        labels = tf.transpose(labels, [1, 0, 2])
        tokens_logits = tf.transpose(tokens_logits, [2, 1, 0, 3])
        tokens = tf.transpose(tokens, [2, 1, 0, 3])
    return (labels_logits, labels, tokens_logits, tokens), states, attention_mask


def strings_output(cell, inputs_states, tokens_states, string_length, num_words, hidden_size):
    state_size = cell.state_size
    batch_size = tokens_states.get_shape()[0].value
    num_conditions = array_ops.shape(tokens_states)[1]
    sequence_length = array_ops.shape(tokens_states)[2]
    tokens_state_size = tokens_states.get_shape()[3].value
    num_strings = num_conditions * sequence_length
    tokens_states = tf.reshape(tokens_states, [batch_size, num_strings, tokens_state_size])
    tokens_states = tf.transpose(tokens_states, [1, 0, 2])

    bias_initializer = init_ops.constant_initializer(0)
    with vs.variable_scope("HiddenLayerVariables"):
        W_state = vs.get_variable(_WEIGHTS_NAME, [tokens_state_size, state_size])
        B_state = vs.get_variable(_BIAS_NAME, [state_size], initializer=bias_initializer)
    with vs.variable_scope("OutputLayerVariables"):
        W_out = vs.get_variable(_WEIGHTS_NAME, [hidden_size, num_words])
        B_out = vs.get_variable(_BIAS_NAME, [num_words], initializer=bias_initializer)
    with vs.variable_scope("Arrays"):
        strings_ta = tf.TensorArray(tf.float32, num_strings, tensor_array_name="strings", clear_after_read=False)

    def time_step(time, strings_ta):
        inputs = tf.zeros([string_length, batch_size, hidden_size])
        initial_state = tf.gather(tokens_states, time)
        initial_state = tf.nn.relu_layer(initial_state, W_state, B_state)
        outputs, states, attention_mask = attention_dynamic_rnn(
            cell, inputs, inputs_states, hidden_size, initial_state, loop_function=lambda output, time: output)
        strings_ta = strings_ta.write(time, outputs)
        return time + 1, strings_ta

    _, strings_ta = control_flow_ops.while_loop(lambda time, *_: time < num_strings, time_step, (0, strings_ta))

    strings = strings_ta.stack()
    strings = tf.reshape(strings, [num_strings * string_length * batch_size, hidden_size])
    strings_logits = tf.nn.relu_layer(strings, W_out, B_out)
    strings = tf.nn.softmax(strings_logits, 1)
    strings_logits = tf.reshape(strings_logits, [num_conditions, sequence_length, string_length, batch_size, num_words])
    strings = tf.reshape(strings, [num_conditions, sequence_length, string_length, batch_size, num_words])
    strings_logits = tf.transpose(strings_logits, [3, 0, 1, 2, 4])
    strings = tf.transpose(strings, [3, 0, 1, 2, 4])
    return strings_logits, strings
