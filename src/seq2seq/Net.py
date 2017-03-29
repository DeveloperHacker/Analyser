import os
import random
import time
from abc import ABCMeta

import tensorflow as tf

from variables.path import SEQ2SEQ
from variables.train import BLOCK_SIZE


class Net(metaclass=ABCMeta):
    @property
    def saver(self):
        if self._saver is None:
            trainable_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope)
            self._saver = tf.train.Saver(var_list=trainable_variables)
        return self._saver

    def __init__(self, name: str):
        self.scope = None
        self._saver = None
        self.name = name
        self.path = None
        self.train_set = None
        self.validation_set = None
        self.postfix = "-net-"

    def get_train_set(self):
        return random.sample(self.train_set, min(BLOCK_SIZE, len(self.train_set)))

    def get_validation_set(self):
        return random.sample(self.validation_set, min(BLOCK_SIZE, len(self.validation_set)))

    def mkdir(self):
        self.path = SEQ2SEQ + "/" + self.name + self.postfix + time.strftime("%d-%m-%Y-%H-%M-%S")
        os.mkdir(self.path)

    def save(self, session: tf.Session):
        if self.path is None:
            self.mkdir()
        self.saver.save(session, self.path + "/model.ckpt")

    def newest(self):
        date = None
        names = []
        for name in os.listdir(SEQ2SEQ):
            if self.name + self.postfix in name:
                names.append(name[len(self.name) + len(self.postfix):])
        max_date = None
        for date in names:
            splitted = [0.0] * 6
            splitted[2], splitted[1], splitted[0], splitted[3], splitted[4], splitted[5] = date.split("-")
            if max_date is None:
                max_date = splitted
            else:
                for a, b in zip(splitted, max_date):
                    if a > b:
                        max_date = date
                        break
                    elif a < b:
                        break
        if date is None:
            raise FileNotFoundError("Saves from {} net is not found".format(self.name))
        return date

    def restore(self, session: tf.Session, date: str = None):
        date = self.newest() if date is None else date
        self.path = "{}/{}{}{}".format(SEQ2SEQ, self.name, self.postfix, date)
        self.saver.restore(session, self.path + "/model.ckpt")
