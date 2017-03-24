import os
import random
import time
from abc import ABCMeta, abstractmethod

import tensorflow as tf

from variables.path import Q_FUNCTION, ANALYSER_MODEL
from variables.train import BLOCK_SIZE


class Net(metaclass=ABCMeta):
    @property
    def saver(self):
        if self._saver is None:
            trainable_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope)
            self._saver = tf.train.Saver(var_list=trainable_variables)
        return self._saver

    def __init__(self):
        self.scope = None
        self._saver = None

    @abstractmethod
    def save(self, session: tf.Session):
        pass

    @abstractmethod
    def restore(self, session: tf.Session):
        pass


class AnalyserNet(Net):
    def __init__(self):
        super().__init__()
        self.inputs = None
        self.inputs_sizes = None

        self.output = None
        self.optimise = None
        self.losses = None

        self.train_set = None
        self.validation_set = None

        self.path = ANALYSER_MODEL

    def save(self, session: tf.Session):
        self.saver.save(session, self.path)

    def restore(self, session: tf.Session):
        self.saver.restore(session, self.path)


class QFunctionNet(Net):
    def __init__(self):
        super().__init__()
        self.inputs = None
        self.inputs_sizes = None
        self.output = None
        self.evaluation = None

        self.optimise = None
        self.q = None

        self.losses = None

        self.train_set = None
        self.validation_set = None

        self.path = None

    def get_train_set(self):
        return random.sample(self.train_set, min(BLOCK_SIZE, len(self.train_set)))

    def get_validation_set(self):
        return random.sample(self.validation_set, min(BLOCK_SIZE, len(self.validation_set)))

    def mkdir(self):
        self.path = Q_FUNCTION + "/q-function-net-{}".format(time.strftime("%d-%m-%Y-%H-%M-%S"))
        os.mkdir(self.path)
        self.path += "/model.ckpt"

    def save(self, session: tf.Session):
        self.saver.save(session, self.path)

    def restore(self, session: tf.Session):
        self.saver.restore(session, self.path)
