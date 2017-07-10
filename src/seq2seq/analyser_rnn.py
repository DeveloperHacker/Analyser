import tensorflow as tf
from tensorflow.python.ops import control_flow_ops, array_ops, variable_scope as vs, init_ops

from seq2seq import dynamic_rnn
from seq2seq.dynamic_rnn import stack_attention_dynamic_rnn, stack_bidirectional_dynamic_rnn

# noinspection PyProtectedMember
_WEIGHTS_NAME = dynamic_rnn._WEIGHTS_NAME
# noinspection PyProtectedMember
_BIAS_NAME = dynamic_rnn._BIAS_NAME


def sequence_input(cells_fw, cells_bw, inputs: list, sequence_lengths: list, dtype):
    if len(inputs) != len(sequence_lengths):
        raise ValueError("Number of inputs and inputs lengths must be equals")
    if len(inputs) == 0:
        raise ValueError("Number of inputs must be greater zero")

    batch_size = None
    for _inputs, _sequence_lengths in zip(inputs, sequence_lengths):
        _batch_size = _inputs.get_shape()[0].value
        if batch_size is not None and _batch_size != batch_size:
            raise ValueError("Batch sizes for any inputs must be equals")
        batch_size = _sequence_lengths.get_shape()[0].value
        if _batch_size != batch_size:
            raise ValueError("Batch sizes of Inputs and inputs lengths must be equals")

    attention_states = []
    for i, (_inputs, _sequence_lengths) in enumerate(zip(inputs, sequence_lengths)):
        with vs.variable_scope("encoder_%d" % i):
            encoder_outputs, states_fw, states_bw = stack_bidirectional_dynamic_rnn(
                cells_fw, cells_bw, _inputs, sequence_length=_sequence_lengths, dtype=dtype)
            attention_states.append(tf.concat((states_fw[-1], states_bw[-1]), 2))
    return attention_states


def input_projection(inputs, projection_size, dtype):
    input_size = None
    for _inputs in inputs:
        _input_size = _inputs.get_shape()[2].value
        if input_size is not None and _input_size != input_size:
            raise ValueError("Input sizes for any inputs must be equals")
        input_size = _input_size

    with vs.variable_scope("input_projection"):
        W_enc = vs.get_variable(_WEIGHTS_NAME, [input_size, projection_size], dtype)
        B_enc = vs.get_variable(_BIAS_NAME, [projection_size], dtype, init_ops.constant_initializer(0, dtype))
        projections = []
        for i, _inputs in enumerate(inputs):
            batch_size = _inputs.get_shape()[0].value
            input_length = _inputs.get_shape()[1].value
            if input_length is None:
                input_length = tf.shape(_inputs)[1]
            _inputs = tf.reshape(_inputs, [batch_size * input_length, input_size])
            projection = _inputs @ W_enc + B_enc
            projection = tf.reshape(projection, [batch_size, input_length, projection_size])
            projections.append(projection)
    return projections


def analysing_loss(targets, logits, variables, l2_loss_weight=0.0001):
    with vs.variable_scope("analysing_loss"):
        data_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=targets, logits=logits)
        data_loss = tf.reduce_mean(data_loss, list(range(1, len(data_loss.shape))))
        l2_loss = l2_loss_weight * tf.reduce_sum([tf.nn.l2_loss(variable) for variable in variables])
        loss = tf.sqrt(tf.square(data_loss) + tf.square(l2_loss))
    return loss


def sequence_output(attention_states,
                    root_cells, root_time_steps, root_num_heads,
                    sequence_cells, sequence_time_steps, sequence_num_heads,
                    output_size,
                    dtype):
    if len(sequence_cells) != len(root_cells):
        raise ValueError("Number of sequence cells and root cells must be equals")
    batch_size = None
    for state in attention_states:
        _batch_size = state.get_shape()[0].value
        if batch_size is not None and _batch_size != batch_size:
            raise ValueError("Batch sizes for any Attention states must be equals")
        batch_size = _batch_size
    with vs.variable_scope("root_decoder"):
        root_decoder_inputs = tf.zeros([root_time_steps, batch_size, output_size], dtype)
        root_decoder_outputs, root_decoder_states = stack_attention_dynamic_rnn(
            root_cells, root_decoder_inputs, attention_states, output_size, root_num_heads, dtype=dtype)
        roots = root_decoder_states[-1]
        num_roots = roots.get_shape()[0].value
        if num_roots is None:
            num_roots = array_ops.shape(roots)[0]
        root_decoder_states = tf.transpose(tf.stack(root_decoder_states), [1, 0, 2, 3])
    with vs.variable_scope("sequence_decoder"):
        time_steps = sequence_time_steps + 1
        with vs.variable_scope("Arrays"):
            outputs_ta = tf.TensorArray(
                dtype,
                size=num_roots,
                tensor_array_name="outputs",
                clear_after_read=False)
            states_ta = tf.TensorArray(
                dtype,
                size=num_roots,
                tensor_array_name="states",
                clear_after_read=False)

        def _time_step(time, _outputs_ta, _states_ta):
            sequence_decoder_inputs = tf.zeros([time_steps, batch_size, output_size], dtype)
            initial_states = tf.unstack(tf.gather(root_decoder_states, time))
            sequence_decoder_outputs, sequence_decoder_states = stack_attention_dynamic_rnn(
                sequence_cells,
                sequence_decoder_inputs,
                attention_states,
                output_size,
                sequence_num_heads,
                initial_states=initial_states,
                dtype=dtype)
            _outputs_ta = _outputs_ta.write(time, sequence_decoder_outputs[-1])
            _states_ta = _states_ta.write(time, sequence_decoder_states[-1])
            return time + 1, _outputs_ta, _states_ta

        _, outputs_ta, states_ta = control_flow_ops.while_loop(
            cond=lambda _time, *_: _time < num_roots, body=_time_step, loop_vars=(0, outputs_ta, states_ta))

        with vs.variable_scope("OutputProjection"):
            outputs_logits = outputs_ta.stack()
            outputs_logits = tf.reshape(outputs_logits, [time_steps * num_roots * batch_size, output_size])
            outputs = tf.nn.softmax(outputs_logits, 1)
            outputs_logits = tf.reshape(outputs_logits, [time_steps, num_roots, batch_size, output_size])
            outputs = tf.reshape(outputs, [time_steps, num_roots, batch_size, output_size])
        outputs_logits = tf.transpose(outputs_logits, [2, 1, 0, 3])
        outputs = tf.transpose(outputs, [2, 1, 0, 3])
    return outputs_logits, outputs


def tree_output(attention_states, root_cells, root_time_steps, root_num_heads, num_tokens, tree_height, dtype):
    batch_size = None
    for state in attention_states:
        _batch_size = state.get_shape()[0].value
        if batch_size is not None and _batch_size != batch_size:
            raise ValueError("Batch sizes for any Attention states must be equals")
        batch_size = _batch_size

    with vs.variable_scope("root_decoder"):
        root_decoder_inputs = tf.zeros([root_time_steps, batch_size, num_tokens], dtype)
        root_decoder_outputs, root_decoder_states = stack_attention_dynamic_rnn(
            root_cells,
            root_decoder_inputs,
            attention_states,
            num_tokens,
            root_num_heads,
            dtype=dtype)
        roots = root_decoder_states[-1]
        labels = root_decoder_outputs
        num_roots = roots.get_shape()[0].value
        if num_roots is None:
            num_roots = array_ops.shape(roots)[0]
        state_size = roots.get_shape()[2].value
        states_ta_size = 2 ** tree_height - 1

    with vs.variable_scope("tree_decoder"):
        bias_initializer = init_ops.constant_initializer(0, dtype)
        with vs.variable_scope("LabelProjectionVariables"):
            W_lbl = vs.get_variable(_WEIGHTS_NAME, [num_tokens, num_tokens], dtype)
            B_lbl = vs.get_variable(_BIAS_NAME, [num_tokens], dtype, bias_initializer)
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
            states_ta = tf.TensorArray(
                dtype,
                size=states_ta_size,
                tensor_array_name="states",
                element_shape=roots.get_shape(),
                clear_after_read=False)

        def _time_step(time, size, _states_ta):
            _state = _states_ta.read(time)
            _state = tf.reshape(_state, [num_roots * batch_size, state_size])
            _left_state = _state @ W_left + B_left
            _right_state = _state @ W_right + B_right
            _left_state = tf.reshape(_left_state, [num_roots, batch_size, state_size])
            _right_state = tf.reshape(_right_state, [num_roots, batch_size, state_size])
            _states_ta = _states_ta.write(size, _left_state)
            _states_ta = _states_ta.write(size + 1, _right_state)
            return time + 1, size + 2, _states_ta

        num_loops = 2 ** (tree_height - 1) - 1
        loop_condition = lambda _time, *_: _time < num_loops
        states_ta = states_ta.write(0, roots)
        _, _, states_ta = control_flow_ops.while_loop(loop_condition, _time_step, (0, 1, states_ta))
        with vs.variable_scope("LabelProjection"):
            labels = tf.reshape(labels, [num_roots * batch_size, num_tokens])
            labels_logits = labels @ W_lbl + B_lbl
            labels = tf.nn.softmax(labels_logits, 1)
            labels_logits = tf.reshape(labels_logits, [1, num_roots, batch_size, num_tokens])
            labels = tf.reshape(labels, [1, num_roots, batch_size, num_tokens])
        with vs.variable_scope("OutputProjection"):
            states = states_ta.stack()
            states = tf.reshape(states, [states_ta_size * num_roots * batch_size, state_size])
            outputs_logits = states @ W_out + B_out
            outputs = tf.nn.softmax(outputs_logits, 1)
            outputs_logits = tf.reshape(outputs_logits, [states_ta_size, num_roots, batch_size, num_tokens])
            outputs = tf.reshape(outputs, [states_ta_size, num_roots, batch_size, num_tokens])
        outputs_logits = tf.concat((labels_logits, outputs_logits), 0)
        outputs = tf.concat((labels, outputs), 0)
        outputs_logits = tf.transpose(outputs_logits, [2, 1, 0, 3])
        outputs = tf.transpose(outputs, [2, 1, 0, 3])
    return outputs_logits, outputs
