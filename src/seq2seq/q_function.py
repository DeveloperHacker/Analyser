from collections import namedtuple

from seq2seq import analyser
from seq2seq.seq2seq import *
from utils.Figure import Figure
from utils.wrapper import *
from variables.embeddings import *
from variables.path import *
from variables.sintax import *
from variables.train import *

Inputs = namedtuple("Inputs", ["inputs", "inputs_sizes", "initial_decoder_state", "output", "evaluation"])
Outputs = namedtuple("Outputs", ["q"])
Fetches = namedtuple("Fetches", ["optimise", "loss"])


@trace
def build_net(inputs: Inputs):
    inputs, inputs_sizes, initial_decoder_state, output, _ = inputs
    with vs.variable_scope("q-function"):
        inputs_states = []
        for label in PARTS:
            with vs.variable_scope(label):
                states_fw, states_bw = build_encoder(inputs[label], INPUT_STATE_SIZE, inputs_sizes[label])
                inputs_states.append(tf.concat(axis=1, values=[states_fw[0], states_bw[-1]]))
                inputs_states.append(tf.concat(axis=1, values=[states_fw[-1], states_bw[0]]))
        with vs.variable_scope("output"):
            # output = tf.transpose(output, [1, 0, 2])
            states_fw, states_bw = build_encoder(output, OUTPUT_STATE_SIZE)
            output_states = [tf.concat(axis=1, values=[states_fw[i], states_bw[-(i + 1)]]) for i in range(OUTPUT_SIZE)]
        inputs_states = tf.transpose(tf.stack(inputs_states), [1, 0, 2])
        Q, _ = build_decoder(output_states, inputs_states, INPUT_STATE_SIZE, initial_decoder_state, 1)
        Q = tf.transpose(tf.reshape(tf.stack(Q), [OUTPUT_SIZE, BATCH_SIZE]), [1, 0])
        with tf.variable_scope("softmax"):
            W_shape = [OUTPUT_STATE_SIZE * 2, 1]
            B_shape = [1]
            W = tf.Variable(initial_value=tf.truncated_normal(W_shape, stddev=0.01, dtype=tf.float32), name="weights")
            B = tf.Variable(initial_value=tf.truncated_normal(B_shape, stddev=0.01, dtype=tf.float32), name="biases")
            tf.summary.histogram('test', W)
            output_states = tf.reshape(tf.stack(output_states), [BATCH_SIZE * OUTPUT_SIZE, OUTPUT_STATE_SIZE * 2])
            I = tf.nn.softmax(tf.reshape(tf.matmul(output_states, W) + B, [BATCH_SIZE, OUTPUT_SIZE]))
            q = tf.reduce_sum(Q * I, axis=1)
    return Outputs(q)


@trace
def build_placeholders():
    inputs = {}
    inputs_sizes = {}
    for label in PARTS:
        with vs.variable_scope(label):
            inputs[label] = []
            for i in range(INPUT_SIZE):
                placeholder = tf.placeholder(tf.float32, [BATCH_SIZE, EMBEDDING_SIZE], "batch_%d" % i)
                inputs[label].append(placeholder)
                inputs_sizes[label] = tf.placeholder(tf.int32, [BATCH_SIZE], "input_sizes")
    with vs.variable_scope("output"):
        output = []
        for i in range(OUTPUT_SIZE):
            placeholder = tf.placeholder(tf.float32, [BATCH_SIZE, NUM_TOKENS], "output_%d" % i)
            output.append(placeholder)
    initial_decoder_state = tf.placeholder(tf.float32, [BATCH_SIZE, INPUT_STATE_SIZE], "initial_decoder_state")
    evaluation = tf.placeholder(tf.float32, [BATCH_SIZE], "evaluation")
    return Inputs(inputs, inputs_sizes, initial_decoder_state, output, evaluation)


def build_fetches(inputs: Inputs, outputs: Outputs):
    with vs.variable_scope("loss"):
        trainable_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "q-function")
        loss = tf.abs(inputs.evaluation - outputs.q)
    with vs.variable_scope("optimiser"):
        optimise = tf.train.AdamOptimizer(beta1=0.90).minimize(loss, var_list=trainable_variables)
    return Fetches(optimise, loss)


@trace
def build_feed_dicts(batches: list, evaluate: callable, inputs: Inputs):
    analyser_inputs = analyser.Inputs(*(inputs[:3]))
    output = analyser.build_net(analyser_inputs)
    feed_dicts = analyser.build_feed_dicts(batches, analyser_inputs)
    with tf.Session() as session, tf.device('/cpu:0'):
        # fixme:
        session.run(tf.global_variables_initializer())
        # saver = tf.train.Saver()
        # saver.restore(session, ANALYSER_MODEL)
        for feed_dict in feed_dicts:
            local_output = session.run(fetches=output.output, feed_dict=feed_dict)
            feed_dict.update({outp: local_output[i] for i, outp in enumerate(inputs.output)})
            feed_dict[inputs.evaluation] = evaluate(inputs.inputs, local_output)
    return feed_dicts


@trace
def build(evaluate: callable):
    inputs = build_placeholders()
    outputs = build_net(inputs)
    fetches = build_fetches(inputs, outputs)
    batches = analyser.build_batches()
    feed_dicts = build_feed_dicts(batches, evaluate, inputs)
    return (fetches, inputs, outputs), feed_dicts


@sigint
@trace
def train(restore: bool = False):
    (fetches, _, _), feed_dicts = build(lambda inputs, output: [1.0] * BATCH_SIZE)
    with tf.Session() as session, tf.device('/cpu:0'), Figure(xauto=True) as figure:
        saver = tf.train.Saver()
        try:
            if restore:
                saver.restore(session, Q_FUNCTION_MODEL)
            else:
                session.run(tf.global_variables_initializer())
            for epoch in range(Q_FUNCTION_EPOCHS):
                losses = [0.0] * (len(fetches) - 1)
                for feed_dict in feed_dicts:
                    _, *local_losses = session.run(fetches=fetches, feed_dict=feed_dict)
                    for i, loss in enumerate(local_losses):
                        losses[i] += loss
                losses = [losses[i] / len(feed_dicts) for i, loss in enumerate(losses)]
                string = " ".join(("%7.3f" % loss for loss in losses))
                logging.info("Epoch: {:4d}/{:-4d} Losses: [{}]".format(epoch, Q_FUNCTION_EPOCHS, string))
                figure.plot(epoch, losses[0])
                if epoch % 50:
                    saver.save(session, Q_FUNCTION_MODEL)
        except SIGINTException:
            logging.error("SIGINT")
        finally:
            saver.save(session, Q_FUNCTION_MODEL)


@sigint
@trace
def test():
    (fetches, _, _), feed_dicts = build(lambda inputs, output: [1.0] * BATCH_SIZE)
    pass
