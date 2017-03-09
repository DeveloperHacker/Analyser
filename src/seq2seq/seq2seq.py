import logging

import numpy as np
import tensorflow as tf

from seq2seq import Q_function
from seq2seq import analyser, fetches
from utils import dumper
from utils.Figure import Figure
from utils.wrapper import trace, sigint, SIGINTException
from variables.embeddings import EMBEDDINGS
from variables.path import *
from variables.sintax import *
from variables.train import *


def closest(point, vectors):
    i = np.argmin([np.linalg.norm(point - vector) for vector in vectors])
    return vectors[i]


@trace
def test():
    embeddings = list(EMBEDDINGS)
    ((_, _LOSS, _), (vars_BATCH, _, _), (_OUTPUTS, _)), feed_dicts = init()
    _FIRST = [None] * 4
    for i, label in enumerate(PARTS):
        _FIRST[i] = vars_BATCH[label][0]
    with tf.Session() as session:
        saver = tf.train.Saver()
        saver.restore(session, SEQ2SEQ_MODEL)
        errors = []
        roundLoss = []
        loss = []
        res_loss = []
        for feed_dict in feed_dicts:
            result = session.run(fetches=[_OUTPUTS, _LOSS] + _FIRST, feed_dict=feed_dict)
            outputs = result[0]
            res_loss.append(result[1])
            targets = result[2:]
            assert len(targets) == 4
            for output, target in zip(outputs[:4], targets):
                for out, tar in zip(output, target):
                    emb = closest(out, embeddings)
                    errors.append(np.linalg.norm(emb - tar) > 1e-6)
                    roundLoss.append(np.linalg.norm(emb - tar))
                    loss.append(np.linalg.norm(out - tar))
        logging.info("Accuracy: {}%".format((1 - np.mean(errors)) * 100))
        logging.info("RoundLoss: {}".format(np.mean(roundLoss)))
        logging.info("Loss: {}".format(np.mean(loss)))
        logging.info("ResLoss: {}".format(np.mean(res_loss)))


@trace
def init():
    # Q_function.Q_function()

    inputs, outputs = analyser.analyser()
    _fetches = fetches.pretrain(outputs[0])
    batches = dumper.load(BATCHES)[INPUT_SIZE]
    _feed_dicts = analyser.feed_dicts(batches, *inputs)
    return (_fetches, inputs, outputs), _feed_dicts


@sigint
@trace
def train(restore: bool = False):
    (_fetches, _, _), _feed_dicts = init()
    with tf.Session() as session, tf.device('/cpu:0'), Figure(xauto=True) as figure:
        saver = tf.train.Saver()
        try:
            if restore:
                saver.restore(session, SEQ2SEQ_MODEL)
            else:
                session.run(tf.global_variables_initializer())
            for epoch in range(SEQ2SEQ_EPOCHS):
                losses = None
                for feed_dict in _feed_dicts:
                    _, *local_losses = session.run(fetches=_fetches, feed_dict=feed_dict)
                    if losses is None:
                        losses = local_losses
                    else:
                        for i, loss in enumerate(local_losses):
                            losses[i] += loss
                for i, loss in enumerate(losses):
                    losses[i] /= len(_feed_dicts)
                string = " ".join(("%7.3f" % loss for loss in losses))
                logging.info("Epoch: {:4d}/{:-4d} Losses: [{}]".format(epoch, SEQ2SEQ_EPOCHS, string))
                if epoch > 10:
                    figure.plot(epoch, losses[0])
                if epoch % 50:
                    saver.save(session, SEQ2SEQ_MODEL)
        except SIGINTException:
            logging.error("SIGINT")
        finally:
            saver.save(session, SEQ2SEQ_MODEL)
