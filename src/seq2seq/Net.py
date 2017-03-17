from abc import ABCMeta

import tensorflow as tf

from variables.path import *


class AnalyserOutputs:
    def __init__(self, output, scope):
        self.output = output
        self.scope = scope

    def flatten(self):
        return self.output, self.scope


class Fetches:
    def __init__(self, optimise, losses):
        self.optimise = optimise
        self.losses = losses

    def flatten(self):
        return (self.optimise,) + self.losses.flatten()


class Losses:
    def __init__(self, loss):
        self.loss = loss

    def flatten(self):
        return self.loss,


class AnalyserLosses(Losses):
    def __init__(self, loss, l2_loss, q):
        super().__init__(loss)
        self.l2_loss = l2_loss
        self.q = q

    def flatten(self):
        return super().flatten() + (self.l2_loss, self.q)


class Inputs:
    def __init__(self, inputs=None, inputs_sizes=None, initial_decoder_state=None, output=None, evaluation=None):
        if inputs_sizes is None:
            inputs_sizes = {}
        if inputs is None:
            inputs = {}
        self.inputs = inputs
        self.inputs_sizes = inputs_sizes
        self.initial_decoder_state = initial_decoder_state
        self.output = output
        self.evaluation = evaluation

    def flatten(self):
        return self.inputs, self.inputs_sizes, self.initial_decoder_state, self.output, self.evaluation


class QFunctionOutputs(AnalyserOutputs):
    def __init__(self, output, scope, q):
        super().__init__(output, scope)
        self.q = q

    def flatten(self):
        return super().flatten() + (self.q,)


class Net(metaclass=ABCMeta):
    @property
    def saver(self):
        if self._saver is None:
            trainable_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.outputs.scope)
            self._saver = tf.train.Saver(var_list=trainable_variables)
        return self._saver

    def __init__(self, path: str, fetches=None, inputs=None, outputs=None, feed_dicts=None):
        self.fetches = fetches
        self.inputs = inputs
        self.outputs = outputs
        self.feed_dicts = feed_dicts
        self._saver = None
        self.path = path

    def flatten(self):
        return self.fetches, self.inputs, self.outputs, self.feed_dicts

    def save(self, session: tf.Session):
        self.saver.save(session, self.path)

    def restore(self, session: tf.Session):
        self.saver.restore(session, self.path)


class AnalyserNet(Net):
    def __init__(
            self,
            fetches: Fetches = None,
            inputs: Inputs = None,
            outputs: AnalyserOutputs = None,
            feed_dicts: list = None
    ):
        super().__init__(ANALYSER_MODEL, fetches, inputs, outputs, feed_dicts)


class QFunctionNet(Net):
    def __init__(
            self,
            fetches: Fetches = None,
            inputs: Inputs = None,
            outputs: QFunctionOutputs = None,
            feed_dicts: list = None
    ):
        super().__init__(Q_FUNCTION_MODEL, fetches, inputs, outputs, feed_dicts)
