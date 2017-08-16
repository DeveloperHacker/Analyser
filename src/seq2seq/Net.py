import os
import re
import time
from abc import ABCMeta
from typing import Iterable

import tensorflow as tf

from utils.wrappers import memoize

time_format = "%d-%m-%Y-%H-%M-%S"
time_pattern = "\d{1,2}-\d{1,2}-\d{4}-\d{1,2}-\d{1,2}-\d{1,2}"
model_pattern = re.compile("model-(%s)\.ckpt\.meta" % time_pattern)


def newest(path: str, filtrator):
    try:
        names = [path + "/" + name for name in os.listdir(path) if filtrator(path, name)]
        names.sort(key=os.path.getmtime, reverse=True)
        return names[0]
    except ValueError:
        raise FileNotFoundError("Saves in dir '%s' hasn't found" % path)


class Net(metaclass=ABCMeta):
    @memoize.read_only_property
    def saver(self) -> tf.train.Saver:
        return tf.train.Saver(var_list=self.variables)

    @property
    def variables(self) -> Iterable[tf.Variable]:
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope)

    def __init__(self, working_directory: str, scope=None):
        save_time = time.strftime(time_format)
        self.save_path = os.path.join(working_directory, "model-" + save_time)
        self.scope = scope

    def save(self, session: tf.Session):
        if not os.path.isdir(self.save_path):
            os.mkdir(self.save_path)
        save_time = time.strftime(time_format)
        model_path = os.path.join(self.save_path, "model-%s.ckpt" % save_time)
        self.saver.save(session, model_path)

    def restore(self, session: tf.Session):
        model_filtrator = lambda path, name: os.path.isfile(path + "/" + name) and re.match(model_pattern, name)
        model_path = newest(self.save_path, model_filtrator)
        model_path = ".".join(model_path.split(".")[:-1])
        self.saver.restore(session, model_path)
