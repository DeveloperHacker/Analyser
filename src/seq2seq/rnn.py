from tensorflow.contrib.rnn.python.ops import core_rnn_cell
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import rnn
from tensorflow.python.ops import rnn_cell_impl
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.util import nest

# noinspection PyProtectedMember
_state_size_with_prefix = rnn_cell_impl._state_size_with_prefix
# noinspection PyProtectedMember
_infer_state_dtype = rnn._infer_state_dtype
# noinspection PyProtectedMember
_reverse_seq = rnn._reverse_seq
# noinspection PyProtectedMember
_rnn_step = rnn._rnn_step


def static_rnn(cell, inputs, initial_state=None, dtype=None,
               sequence_length=None, scope=None):
    """Creates a recurrent neural network specified by RNNCell `cell`.

    The simplest form of RNN network generated is:

      ```python
        state = cell.zero_state(...)
        outputs = []
        for input_ in inputs:
          output, state = cell(input_, state)
          outputs.append(output)
        return (outputs, state)
      ```
      However, a few other options are available:

      An initial state can be provided.
      If the sequence_length vector is provided, dynamic calculation is performed.
      This method of calculation does not compute the RNN steps past the maximum
      sequence length of the minibatch (thus saving computational time),
      and properly propagates the state at an example's sequence length
      to the final state output.

      The dynamic calculation performed is, at time `t` for batch row `b`,

      ```python
        (output, state)(b, t) =
          (t >= sequence_length(b))
            ? (zeros(cell.output_size), states(b, sequence_length(b) - 1))
            : cell(input(b, t), state(b, t - 1))
      ```

      Args:
        cell: An instance of RNNCell.
        inputs: A length T list of inputs, each a `Tensor` of shape
          `[batch_size, input_size]`, or a nested tuple of such elements.
        initial_state: (optional) An initial state for the RNN.
          If `cell.state_size` is an integer, this must be
          a `Tensor` of appropriate type and shape `[batch_size, cell.state_size]`.
          If `cell.state_size` is a tuple, this should be a tuple of
          tensors having shapes `[batch_size, s] for s in cell.state_size`.
        dtype: (optional) The data type for the initial state and expected output.
          Required if initial_state is not provided or RNN state has a heterogeneous
          dtype.
        sequence_length: Specifies the length of each sequence in inputs.
          An int32 or int64 vector (tensor) size `[batch_size]`, values in `[0, T)`.
        scope: VariableScope for the created subgraph; defaults to "rnn".

      Returns:
        A pair (outputs, state) where:

        - outputs is a length T list of outputs (one for each input), or a nested
          tuple of such elements.
        - state is the final state

      Raises:
        TypeError: If `cell` is not an instance of RNNCell.
        ValueError: If `inputs` is `None` or an empty list, or if the input depth
          (column size) cannot be inferred from inputs via shape inference.
      """

    if not isinstance(cell, core_rnn_cell.RNNCell):
        raise TypeError("cell must be an instance of RNNCell")
    if not nest.is_sequence(inputs):
        raise TypeError("inputs must be a sequence")
    if not inputs:
        raise ValueError("inputs must not be empty")

    outputs = []
    states = []
    # Create a new scope in which the caching device is either
    # determined by the parent scope, or is set to place the cached
    # Variable using the same placement as for the rest of the RNN.
    with vs.variable_scope(scope or "rnn") as varscope:
        if varscope.caching_device is None:
            varscope.set_caching_device(lambda op: op.device)

        # Obtain the first sequence of the input
        first_input = inputs
        while nest.is_sequence(first_input):
            first_input = first_input[0]

        # Temporarily avoid EmbeddingWrapper and seq2seq badness
        # TODO(lukaszkaiser): remove EmbeddingWrapper
        if first_input.get_shape().ndims != 1:

            input_shape = first_input.get_shape().with_rank_at_least(2)
            fixed_batch_size = input_shape[0]

            flat_inputs = nest.flatten(inputs)
            for flat_input in flat_inputs:
                input_shape = flat_input.get_shape().with_rank_at_least(2)
                batch_size, input_size = input_shape[0], input_shape[1:]
                fixed_batch_size.merge_with(batch_size)
                for i, size in enumerate(input_size):
                    if size.value is None:
                        raise ValueError(
                            "Input size (dimension %d of inputs) must be accessible via "
                            "shape inference, but saw value None." % i)
        else:
            fixed_batch_size = first_input.get_shape().with_rank_at_least(1)[0]

        if fixed_batch_size.value:
            batch_size = fixed_batch_size.value
        else:
            batch_size = array_ops.shape(first_input)[0]
        if initial_state is not None:
            state = initial_state
        else:
            if not dtype:
                raise ValueError("If no initial_state is provided, "
                                 "dtype must be specified")
            state = cell.zero_state(batch_size, dtype)

        if sequence_length is not None:  # Prepare variables
            sequence_length = ops.convert_to_tensor(
                sequence_length, name="sequence_length")
            if sequence_length.get_shape().ndims not in (None, 1):
                raise ValueError(
                    "sequence_length must be a vector of length batch_size")

            def _create_zero_output(output_size):
                # convert int to TensorShape if necessary
                size = _state_size_with_prefix(output_size, prefix=[batch_size])
                output = array_ops.zeros(
                    array_ops.stack(size), _infer_state_dtype(dtype, state))
                shape = _state_size_with_prefix(
                    output_size, prefix=[fixed_batch_size.value])
                output.set_shape(tensor_shape.TensorShape(shape))
                return output

            output_size = cell.output_size
            flat_output_size = nest.flatten(output_size)
            flat_zero_output = tuple(
                _create_zero_output(size) for size in flat_output_size)
            zero_output = nest.pack_sequence_as(structure=output_size,
                                                flat_sequence=flat_zero_output)

            sequence_length = math_ops.to_int32(sequence_length)
            min_sequence_length = math_ops.reduce_min(sequence_length)
            max_sequence_length = math_ops.reduce_max(sequence_length)

        for time, input_ in enumerate(inputs):
            if time > 0: varscope.reuse_variables()
            call_cell = lambda: cell(input_, state)
            if sequence_length is not None:
                # noinspection PyUnboundLocalVariable
                (output, state) = _rnn_step(
                    time=time,
                    sequence_length=sequence_length,
                    min_sequence_length=min_sequence_length,
                    max_sequence_length=max_sequence_length,
                    zero_output=zero_output,
                    state=state,
                    call_cell=call_cell,
                    state_size=cell.state_size)
            else:
                (output, state) = call_cell()

            outputs.append(output)
            states.append(state)

        return outputs, states


def static_bidirectional_rnn(cell_fw, cell_bw, inputs,
                             initial_state_fw=None, initial_state_bw=None,
                             dtype=None, sequence_length=None, scope=None):
    """Creates a bidirectional recurrent neural network.

    Similar to the unidirectional case above (rnn) but takes input and builds
    independent forward and backward RNNs with the final forward and backward
    outputs depth-concatenated, such that the output will have the format
    [time][batch][cell_fw.output_size + cell_bw.output_size]. The input_size of
    forward and backward cell must match. The initial state for both directions
    is zero by default (but can be set optionally) and no intermediate states are
    ever returned -- the network is fully unrolled for the given (passed in)
    length(s) of the sequence(s) or completely unrolled if length(s) is not given.

    Args:
      cell_fw: An instance of RNNCell, to be used for forward direction.
      cell_bw: An instance of RNNCell, to be used for backward direction.
      inputs: A length T list of inputs, each a tensor of shape
        [batch_size, input_size], or a nested tuple of such elements.
      initial_state_fw: (optional) An initial state for the forward RNN.
        This must be a tensor of appropriate type and shape
        `[batch_size, cell_fw.state_size]`.
        If `cell_fw.state_size` is a tuple, this should be a tuple of
        tensors having shapes `[batch_size, s] for s in cell_fw.state_size`.
      initial_state_bw: (optional) Same as for `initial_state_fw`, but using
        the corresponding properties of `cell_bw`.
      dtype: (optional) The data type for the initial state.  Required if
        either of the initial states are not provided.
      sequence_length: (optional) An int32/int64 vector, size `[batch_size]`,
        containing the actual lengths for each of the sequences.
      scope: VariableScope for the created subgraph; defaults to
        "bidirectional_rnn"

    Returns:
      A tuple (outputs, states_fw, states_bw) where:
        outputs is a length `T` list of outputs (one for each input), which
          are depth-concatenated forward and backward outputs.
        states_fw is the final state of the forward rnn.
        states_bw is the final state of the backward rnn.

    Raises:
      TypeError: If `cell_fw` or `cell_bw` is not an instance of `RNNCell`.
      ValueError: If inputs is None or an empty list.
    """

    if not isinstance(cell_fw, core_rnn_cell.RNNCell):
        raise TypeError("cell_fw must be an instance of RNNCell")
    if not isinstance(cell_bw, core_rnn_cell.RNNCell):
        raise TypeError("cell_bw must be an instance of RNNCell")
    if not nest.is_sequence(inputs):
        raise TypeError("inputs must be a sequence")
    if not inputs:
        raise ValueError("inputs must not be empty")

    with vs.variable_scope(scope or "bidirectional_rnn"):
        # Forward direction
        with vs.variable_scope("fw") as fw_scope:
            output_fw, states_fw = static_rnn(
                cell_fw, inputs, initial_state_fw, dtype,
                sequence_length, scope=fw_scope)

        # Backward direction
        with vs.variable_scope("bw") as bw_scope:
            reversed_inputs = _reverse_seq(inputs, sequence_length)
            tmp, states_bw = static_rnn(
                cell_bw, reversed_inputs, initial_state_bw,
                dtype, sequence_length, scope=bw_scope)

    output_bw = _reverse_seq(tmp, sequence_length)
    # Concat each of the forward/backward outputs
    flat_output_fw = nest.flatten(output_fw)
    flat_output_bw = nest.flatten(output_bw)

    flat_outputs = tuple(
        array_ops.concat([fw, bw], 1)
        for fw, bw in zip(flat_output_fw, flat_output_bw))

    outputs = nest.pack_sequence_as(structure=output_fw,
                                    flat_sequence=flat_outputs)

    return outputs, states_fw, states_bw
