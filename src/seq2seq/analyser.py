import logging

import tensorflow as tf
from tensorflow.python.ops import variable_scope as vs

from seq2seq.Q_function import build_q_function
from seq2seq.seq2seq import build_encoder, build_decoder
from utils import dumper, batcher
from utils.Figure import Figure
from utils.batcher import build_butches
from utils.wrapper import trace, sigint, SIGINTException
from variables.embeddings import *
from variables.embeddings import EMBEDDINGS
from variables.path import *
from variables.sintax import PARTS
from variables.train import *


@trace
def build_analyser(inputs, inputs_sizes, initial_decoder_state):
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
            output = tf.nn.softmax(tf.reshape(tf.matmul(output, W) + B, [BATCH_SIZE, OUTPUT_SIZE, NUM_TOKENS]))
    return output


@trace
def build_feed_dicts(batches, inputs, inputs_sizes, initial_decoder_state):
    # noinspection PyShadowingNames
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
def test():
    embeddings = list(EMBEDDINGS)
    # noinspection PyShadowingNames
    ((_, loss, *_), (batch, _, _), output), feed_dicts = build()
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
    return inputs, inputs_sizes, initial_decoder_state


@trace
def build_l2_loss(trainable_variables, regularisation_variable_names):
    with vs.variable_scope("l2_loss"):
        variables = [variable for variable in trainable_variables if variable.name in regularisation_variable_names]
        assert len(variables) == len(regularisation_variable_names)
        l2_loss = tf.reduce_sum([tf.nn.l2_loss(variable) for variable in variables])
    return l2_loss


@trace
def build_fetches(result):
    with vs.variable_scope("loss"):
        trainable_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "analyser")
        l2_loss = build_l2_loss(trainable_variables, ANALYSER_REGULARIZATION_VARIABLES)
        loss = tf.sqrt(tf.square(Q_WEIGHT * result) + tf.square(L2_WEIGHT * l2_loss))
    with vs.variable_scope("optimiser"):
        optimise = tf.train.AdamOptimizer(beta1=0.90).minimize(loss, var_list=trainable_variables)
    return optimise, (loss, l2_loss, result)


@trace
def build():
    inputs, inputs_sizes, initial_decoder_state = build_placeholders()
    output = build_analyser(inputs, inputs_sizes, initial_decoder_state)
    result = build_q_function(inputs, inputs_sizes, initial_decoder_state, output)
    fetches = build_fetches(result)

    with tf.Session() as session:
        summary_writer = tf.summary.FileWriter(ANALYSER, session.graph)
        session.run(tf.global_variables_initializer())
        summary_writer.close()

    exit(1)

    methods = dumper.load(VEC_METHODS)
    baskets = batcher.throwing(methods, [INPUT_SIZE])
    batches = {basket: build_butches(data, BATCH_SIZE) for basket, data in baskets.items()}
    feed_dicts = build_feed_dicts(batches[INPUT_SIZE], *inputs)
    return (fetches, inputs, output), feed_dicts


@sigint
@trace
def train(restore: bool = False):
    # noinspection PyShadowingNames
    (fetches, _, _), feed_dicts = build()
    with tf.Session() as session, tf.device('/cpu:0'), Figure(xauto=True) as figure:
        saver = tf.train.Saver()
        try:
            if restore:
                saver.restore(session, ANALYSER_MODEL)
            else:
                session.run(tf.global_variables_initializer())
            saver.restore(session, Q_FUNCTION_MODEL)
            for epoch in range(SEQ2SEQ_EPOCHS):
                losses = [0.0] * (len(fetches) - 1)
                for feed_dict in feed_dicts:
                    _, *local_losses = session.run(fetches=fetches, feed_dict=feed_dict)
                    for i, loss in enumerate(local_losses):
                        losses[i] += loss
                for i, loss in enumerate(losses):
                    losses[i] /= len(feed_dicts)
                string = " ".join(("%7.3f" % loss for loss in losses))
                logging.info("Epoch: {:4d}/{:-4d} Losses: [{}]".format(epoch, SEQ2SEQ_EPOCHS, string))
                figure.plot(epoch, losses[0])
                if epoch % 50:
                    saver.save(session, ANALYSER_MODEL)
        except SIGINTException:
            logging.error("SIGINT")
        finally:
            saver.save(session, ANALYSER_MODEL)
