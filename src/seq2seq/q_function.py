from seq2seq import analyser
from seq2seq import contract
from seq2seq.Net import *
from seq2seq.seq2seq import *
from utils.Figure import Figure
from utils.handlers import SIGINTException
from utils.wrapper import *
from variables.tags import *
from variables.train import *


@trace
def build_placeholders() -> Inputs:
    inputs = {}
    inputs_sizes = {}
    for label in PARTS:
        with vs.variable_scope(label):
            inputs[label] = []
            for i in range(INPUT_SIZE):
                placeholder = tf.placeholder(tf.float32, [BATCH_SIZE, EMBEDDING_SIZE], "batch_%d" % i)
                inputs[label].append(placeholder)
                inputs_sizes[label] = tf.placeholder(tf.int32, [BATCH_SIZE], "input_sizes")
    initial_decoder_state = tf.placeholder(tf.float32, [BATCH_SIZE, INPUT_STATE_SIZE], "initial_decoder_state")
    evaluation = tf.placeholder(tf.float32, [BATCH_SIZE], "evaluation")
    return Inputs(inputs, inputs_sizes, initial_decoder_state, None, evaluation)


@trace
def build_net(inputs: Inputs) -> QFunctionNet:
    net = QFunctionNet(inputs=inputs)
    inputs, inputs_sizes, initial_decoder_state, output, _ = net.inputs.flatten()
    with vs.variable_scope("q-function"):
        inputs_states = []
        for label in PARTS:
            with vs.variable_scope(label):
                states_fw, states_bw = build_encoder(inputs[label], INPUT_STATE_SIZE, inputs_sizes[label])
                inputs_states.append(tf.concat(axis=1, values=[states_fw[0], states_bw[-1]]))
                inputs_states.append(tf.concat(axis=1, values=[states_fw[-1], states_bw[0]]))
        with vs.variable_scope("output"):
            states_fw, states_bw = build_encoder(output, OUTPUT_STATE_SIZE)
            output_states = [tf.concat(axis=1, values=[states_fw[i], states_bw[-(i + 1)]]) for i in range(OUTPUT_SIZE)]
        inputs_states = tf.transpose(tf.stack(inputs_states), [1, 0, 2])
        Q, _ = build_decoder(output_states, inputs_states, INPUT_STATE_SIZE, initial_decoder_state, 1)
        Q = tf.nn.relu(Q)
        Q = tf.transpose(tf.reshape(tf.stack(Q), [OUTPUT_SIZE, BATCH_SIZE]), [1, 0])
        with vs.variable_scope("softmax"):
            W_shape = [OUTPUT_STATE_SIZE * 2, 1]
            B_shape = [1]
            W = tf.Variable(initial_value=tf.truncated_normal(W_shape, stddev=0.01, dtype=tf.float32), name="weights")
            B = tf.Variable(initial_value=tf.truncated_normal(B_shape, stddev=0.01, dtype=tf.float32), name="biases")
            tf.summary.histogram('test', W)
            output_states = tf.reshape(tf.stack(output_states), [BATCH_SIZE * OUTPUT_SIZE, OUTPUT_STATE_SIZE * 2])
            I = tf.nn.softmax(tf.reshape(tf.matmul(output_states, W) + B, [BATCH_SIZE, OUTPUT_SIZE]))
            q = tf.reduce_sum(Q * I, axis=1)
        scope = vs.get_variable_scope().name
    net.outputs = QFunctionOutputs(output, scope, q)
    return net


@trace
def build_fetches(net: QFunctionNet) -> Fetches:
    scope = vs.get_variable_scope().name
    scope = ("" if len(scope) == 0 else scope + "/") + "q-function"
    with vs.variable_scope("loss"):
        trainable_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope)
        loss = tf.reduce_mean(tf.square(net.inputs.evaluation - net.outputs.q))
        print(loss.shape)
    with vs.variable_scope("optimiser"):
        optimise = tf.train.AdamOptimizer(beta1=0.90).minimize(loss, var_list=trainable_variables)
    return Fetches(optimise, Losses(loss))


@trace
def build_feed_dicts(feed_dicts: list, net: AnalyserNet, evaluate: callable) -> list:
    with tf.Session() as session, tf.device('/cpu:0'):
        session.run(tf.global_variables_initializer())
        net.restore(session)
        for feed_dict in feed_dicts:
            _inputs, _output = session.run(fetches=(net.inputs.inputs, net.inputs.output), feed_dict=feed_dict)
            feed_dict.update({output: _output[i] for i, output in enumerate(net.inputs.output)})
            feed_dict[net.inputs.evaluation] = evaluate(_inputs, _output)
    return feed_dicts


@trace
def build() -> Net:
    inputs = build_placeholders()
    analyser_net = analyser.build_net(inputs)
    analyser_net.inputs.output = analyser_net.outputs.output
    q_function_net = build_net(analyser_net.inputs)
    q_function_net.fetches = build_fetches(q_function_net)
    batches = analyser.build_batches()
    analyser_net.feed_dicts = analyser.build_feed_dicts(analyser_net, batches)
    evaluate = contract.build_evaluator()
    q_function_net.feed_dicts = build_feed_dicts(analyser_net.feed_dicts, analyser_net, evaluate)
    return q_function_net


@trace
def train(net: Net, restore: bool = False):
    with tf.Session() as session, tf.device('/cpu:0'), Figure(xauto=True) as figure:
        writer = tf.summary.FileWriter(SEQ2SEQ, session.graph)
        writer.close()
        exit(1)
        try:
            session.run(tf.global_variables_initializer())
            if restore:
                net.restore(session)
            for epoch in range(Q_FUNCTION_EPOCHS):
                losses = (0.0 for _ in range(len(net.fetches.flatten()) - 1))
                for feed_dict in net.feed_dicts:
                    _, *local_losses = session.run(fetches=net.fetches.flatten(), feed_dict=feed_dict)
                    losses = (local_losses[i] + loss for i, loss in enumerate(losses))
                losses = (loss / len(net.feed_dicts) for loss in losses)
                loss = next(losses)
                logging.info("Epoch: {:4d}/{:-4d} Loss: {:.4f}".format(epoch, Q_FUNCTION_EPOCHS, loss))
                figure.plot(epoch, loss)
                if epoch % 50:
                    net.save(session)
        except SIGINTException:
            logging.error("SIGINT")
        finally:
            net.save(session)


@trace
def test(net: Net):
    pass


@trace
def run(foo: str):
    net = build()
    if foo == "train":
        train(net)
    elif foo == "restore":
        train(net, True)
    elif foo == "test":
        test(net)
