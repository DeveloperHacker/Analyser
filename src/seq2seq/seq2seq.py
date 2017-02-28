import logging

import numpy as np
import tensorflow as tf

from seq2seq.constructor import buildRNN, buildFeedDicts
from utils import dumper, filter
from utils.Figure import Figure
from utils.wrapper import trace, sigint, SIGINTException
from variables import EMBEDDINGS, SEQ2SEQ_MODEL, BATCHES, MAX_ENCODE_SEQUENCE, SEQ2SEQ_EPOCHS


@trace
def train():
    (fetches, _, _), feed_dicts = init()
    _train(fetches, feed_dicts, False)


@trace
def restore():
    (fetches, _, _), feed_dicts = init()
    _train(fetches, feed_dicts, True)


@trace
def test():
    ((_, loss, _), (vars_BATCH, _, _), res_OUTPUTS), feed_dicts = init()
    _test(vars_BATCH, loss, res_OUTPUTS, feed_dicts)


@trace
def _test(vars_BATCH, res_Loss, res_OUTPUTS, feed_dicts):
    embeddings = dumper.load(EMBEDDINGS)
    embeddings = list(embeddings.values())
    vars_FIRST = [None] * 4
    for label, index in filter.parts.items():
        vars_FIRST[index] = vars_BATCH[label][0]
    with tf.Session() as session:
        saver = tf.train.Saver()
        saver.restore(session, SEQ2SEQ_MODEL)
        errors = []
        roundLoss = []
        loss = []
        res_loss = []
        for feed_dict in feed_dicts:
            result = session.run(fetches=[res_OUTPUTS, res_Loss] + vars_FIRST, feed_dict=feed_dict)
            outputs = result[0]
            res_loss.append(result[1])
            targets = result[2:]
            assert len(targets) == 4
            for output, target in zip(outputs[:4], targets):
                for out, tar in zip(output, target):
                    i = np.argmin([np.linalg.norm(out - emb) for emb in embeddings])
                    closest = embeddings[i]
                    errors.append(np.linalg.norm(closest - tar) > 1e-6)
                    roundLoss.append(np.linalg.norm(closest - tar))
                    loss.append(np.linalg.norm(out - tar))
        logging.info("Accuracy: {}%".format((1 - np.mean(errors)) * 100))
        logging.info("RoundLoss: {}".format(np.mean(roundLoss)))
        logging.info("Loss: {}".format(np.mean(loss)))
        logging.info("ResLoss: {}".format(np.mean(res_loss)))


@trace
def init():
    fetches, variables, results = buildRNN(filter.parts)
    batches = dumper.load(BATCHES)[MAX_ENCODE_SEQUENCE]
    feed_dicts = buildFeedDicts(batches, *variables)
    return (fetches, variables, results), feed_dicts


@sigint
def _train(fetches: tuple, feed_dicts: list, restore: bool = False):
    with tf.Session() as session, Figure(xauto=True) as figure:
        saver = tf.train.Saver()
        try:
            if restore:
                saver.restore(session, SEQ2SEQ_MODEL)
            else:
                session.run(tf.global_variables_initializer())
            for epoch in range(SEQ2SEQ_EPOCHS):
                loss = 0
                l2_loss = 0
                for feed_dict in feed_dicts:
                    _, local_loss, local_l2_loss = session.run(fetches=fetches, feed_dict=feed_dict)
                    loss += local_loss
                    l2_loss += local_l2_loss
                loss /= len(feed_dicts)
                l2_loss /= len(feed_dicts)
                figure.plot(epoch, loss)
                logging.info("Epoch: %4d/%-4d; Loss: %5.4f; L2 loss: %5.4f" % (epoch, SEQ2SEQ_EPOCHS, loss, l2_loss))
                if epoch % 50:
                    saver.save(session, SEQ2SEQ_MODEL)
        except SIGINTException:
            logging.error("SIGINT")
        finally:
            saver.save(session, SEQ2SEQ_MODEL)
