import os
import time
from abc import ABCMeta

import tensorflow as tf

from variables.paths import SEQ2SEQ
from variables.train import BLOCK_SIZE


class Net(metaclass=ABCMeta):
    @staticmethod
    def get_part(data_set, ptr, step):
        if ptr >= len(data_set):
            ptr = 0
        left = ptr
        right = min(len(data_set), left + step)
        ptr = 0 if right == len(data_set) else ptr + step
        return data_set[left: right], ptr

    def get_train_set(self):
        data_set, self.train_set_ptr = Net.get_part(self.train_set, self.train_set_ptr, BLOCK_SIZE)
        return data_set

    def get_validation_set(self):
        data_set, self.validation_set_ptr = Net.get_part(self.validation_set, self.validation_set_ptr, BLOCK_SIZE)
        return data_set

    def get_optimiser(self):
        optimiser = self.optimisers[self.optimisers_prt]
        self.optimisers_prt = (self.optimisers_prt + 1) % len(self.optimisers)
        return optimiser

    def get_saver(self):
        if self._saver is None:
            self._saver = tf.train.Saver(var_list=self.get_variables())
        return self._saver

    def get_variables(self):
        if self._variables is None:
            self._variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope)
        return self._variables

    def __init__(self, name: str):
        self.scope = None
        self._variables = None
        self._saver = None
        self.name = name + "-net-"
        self.path = None
        self.train_set = None
        self.validation_set = None
        self.train_set_ptr = 0
        self.validation_set_ptr = 0
        self.optimisers = None
        self.optimisers_prt = 0

    def reset(self, session: tf.Session):
        self.path = None
        self._saver = None
        session.run(tf.global_variables_initializer())

    def mkdir(self):
        self.path = SEQ2SEQ + "/" + self.name + time.strftime("%d-%m-%Y-%H-%M-%S")
        os.mkdir(self.path)

    def save(self, session: tf.Session):
        if self.path is None:
            self.mkdir()
        self.get_saver().save(session, self.path + "/model-{}.ckpt".format(time.strftime("%d-%m-%Y-%H-%M-%S")))

    def newest(self, path: str, filtrator):
        names = []
        for name in os.listdir(path):
            pathname = os.path.join(path, name)
            if filtrator(pathname):
                names.append(pathname)
        if len(names) == 0:
            raise FileNotFoundError("Saves from {} net is not found".format(self.name))
        names.sort(key=os.path.getmtime, reverse=True)
        last = names[0]
        return last

    def restore(self, session: tf.Session):
        filtrator = lambda path: os.path.isdir(path) and len(os.listdir(path)) > 0 and self.name in path
        self.path = self.newest(SEQ2SEQ, filtrator)
        import re
        model_filtrator = lambda path: os.path.isfile(path) and re.match(r".+/model-.+\.ckpt\..+", path)
        model = self.newest(self.path, model_filtrator)
        model = ".".join(model.split(".")[:-1])
        self.get_saver().restore(session, model)
