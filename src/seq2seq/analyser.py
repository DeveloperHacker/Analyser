from collections import namedtuple

from seq2seq import q_function
from seq2seq.seq2seq import *
from utils import batcher, dumper
from utils.Figure import Figure
from utils.handlers import SIGINTException
from utils.wrapper import *
from variables.embeddings import *
from variables.path import *
from variables.tags import *
from variables.train import *

Inputs = namedtuple("Inputs", ["inputs", "inputs_sizes", "initial_decoder_state"])
Outputs = namedtuple("Outputs", ["output"])
Fetches = namedtuple("Fetches", ["optimise", "loss", "l2_loss", "q"])


@trace
def build_net(inputs: Inputs):
    inputs, inputs_sizes, initial_decoder_state = inputs
    with vs.variable_scope("analyser"):
        inputs_states = []
        for label in PARTS:
            with vs.variable_scope(label):
                output_states_fw, output_states_bw = build_encoder(inputs[label], INPUT_STATE_SIZE, inputs_sizes[label])
                inputs_states.append(tf.concat(axis=1, values=[output_states_fw[0], output_states_bw[-1]]))
                inputs_states.append(tf.concat(axis=1, values=[output_states_fw[-1], output_states_bw[0]]))
        goes = tf.stack([GO for _ in range(BATCH_SIZE)])
        decoder_inputs = [goes for _ in range(OUTPUT_SIZE)]
        inputs_states = tf.stack(inputs_states)
        inputs_states = tf.transpose(inputs_states, [1, 0, 2])
        output, _ = build_decoder(decoder_inputs, inputs_states, INPUT_STATE_SIZE, initial_decoder_state,
                                  EMBEDDING_SIZE, loop=True)
        with tf.variable_scope("softmax"):
            W_shape = [EMBEDDING_SIZE, NUM_TOKENS]
            B_shape = [NUM_TOKENS]
            W = tf.Variable(initial_value=tf.truncated_normal(W_shape, stddev=0.01, dtype=tf.float32), name="weights")
            B = tf.Variable(initial_value=tf.truncated_normal(B_shape, stddev=0.01, dtype=tf.float32), name="biases")
            output = tf.reshape(tf.stack(output), [BATCH_SIZE * OUTPUT_SIZE, EMBEDDING_SIZE])
            output = tf.nn.softmax(tf.reshape(tf.matmul(output, W) + B, [OUTPUT_SIZE, BATCH_SIZE, NUM_TOKENS]))
            output = tf.unstack(output)
    return Outputs(output)


@trace
def build_feed_dicts(batches: list, inputs: Inputs):
    inputs, inputs_sizes, initial_decoder_state = inputs
    feed_dicts = []
    for batch in batches:
        feed_dict = {}
        for label, (lines, _) in batch.items():
            for _LINE in inputs[label]:
                feed_dict[_LINE] = []
            for embeddings in lines:
                line = embeddings + [PAD for _ in range(INPUT_SIZE - len(embeddings))]
                for i, embedding in enumerate(line):
                    feed_dict[inputs[label][i]].append(embedding)
            feed_dict[inputs_sizes[label]] = tuple(len(emb) for emb in lines)
        feed_dict[initial_decoder_state] = INITIAL_STATE
        feed_dicts.append(feed_dict)
    return feed_dicts


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
    initial_decoder_state = tf.placeholder(tf.float32, [BATCH_SIZE, INPUT_STATE_SIZE], "initial_decoder_state")
    return Inputs(inputs, inputs_sizes, initial_decoder_state)


@trace
def build_fetches(outputs: q_function.Outputs):
    with vs.variable_scope("loss"):
        trainable_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "analyser")
        l2_loss = build_l2_loss(trainable_variables, ANALYSER_REGULARIZATION_VARIABLES)
        q = tf.reduce_mean(outputs.q)
        loss = Q_WEIGHT * q + L2_WEIGHT * l2_loss
    with vs.variable_scope("optimiser"):
        optimise = tf.train.AdamOptimizer(beta1=0.90).minimize(loss, var_list=trainable_variables)
    return Fetches(optimise, loss, l2_loss, q)


@trace
def build_batches():
    methods = dumper.load(VEC_METHODS)
    baskets = batcher.throwing(methods, [INPUT_SIZE])
    batches = {basket: batcher.build_batches(data[:BATCH_SIZE * 2], BATCH_SIZE) for basket, data in baskets.items()}
    return batches[INPUT_SIZE]


@trace
def build():
    inputs = build_placeholders()
    outputs = build_net(inputs)
    q_function_inputs = q_function.Inputs(*inputs, outputs.output, None)
    q_function_outputs = q_function.build_net(q_function_inputs)
    fetches = build_fetches(q_function_outputs)
    batches = build_batches()
    feed_dicts = build_feed_dicts(batches, inputs)
    return (fetches, inputs, outputs), feed_dicts


@trace
def build_saver() -> tf.train.Saver:
    trainable_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "analyser")
    saver = tf.train.Saver(var_list=trainable_variables)
    return saver


@trace
def pretrain():
    pass


@trace
def train(restore: bool = False):
    (fetches, _, _), feed_dicts = build()
    with tf.Session() as session, tf.device('/cpu:0'), Figure(xauto=True) as figure:
        saver = None
        try:
            session.run(tf.global_variables_initializer())
            saver = q_function.build_saver()
            saver.restore(session, Q_FUNCTION_MODEL)
            saver = build_saver()
            if restore:
                saver.restore(session, ANALYSER_MODEL)
            for epoch in range(ANALYSER_EPOCHS):
                losses = (0.0 for _ in range(len(fetches) - 1))
                for feed_dict in feed_dicts:
                    _, *local_losses = session.run(fetches=fetches, feed_dict=feed_dict)
                    losses = (local_losses[i] + loss for i, loss in enumerate(losses))
                losses = (loss / len(feed_dicts) for loss in losses)
                loss = next(losses)
                l2_loss = next(losses)
                q = next(losses)
                logging.info("Epoch: {:4d}/{:-4d} Loss: {:.4f} L2Loss: {:.4f} Q: {:.4f}"
                             .format(epoch, ANALYSER_EPOCHS, loss, l2_loss, q))
                figure.plot(epoch, loss)
                if epoch % 50:
                    saver.save(session, ANALYSER_MODEL)
        except SIGINTException:
            logging.error("SIGINT")
        finally:
            if saver is None:
                saver = build_saver()
            saver.save(session, ANALYSER_MODEL)


@trace
def test():
    embeddings = list(EMBEDDINGS)
    ((_, loss, _, _), (batch, _, _), output), feed_dicts = build()
    first = [batch[label][0] for label in PARTS]
    with tf.Session() as session:
        saver = tf.train.Saver()
        saver.restore(session, ANALYSER_MODEL)
        errors = []
        roundLoss = []
        loss = []
        res_loss = []
        for _feed_dict in feed_dicts:
            result = session.run(fetches=[output, loss] + first, feed_dict=_feed_dict)
            outputs = result[0]
            res_loss.append(result[1])
            targets = result[2:]
            assert len(targets) == 4
            for output, target in zip(outputs[:4], targets):
                for out, tar in zip(output, target):
                    i = np.argmin([np.linalg.norm(out - embedding) for embedding in embeddings])
                    errors.append(np.linalg.norm(embeddings[i] - tar) > 1e-6)
                    roundLoss.append(np.linalg.norm(embeddings[i] - tar))
                    loss.append(np.linalg.norm(out - tar))
        logging.info("Accuracy: {}%".format((1 - np.mean(errors)) * 100))
        logging.info("RoundLoss: {}".format(np.mean(roundLoss)))
        logging.info("Loss: {}".format(np.mean(loss)))
        logging.info("ResLoss: {}".format(np.mean(res_loss)))
