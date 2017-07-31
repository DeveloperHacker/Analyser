import os
import re
import time
from abc import ABCMeta
from typing import Iterable

import tensorflow as tf

from utils.wrappers import lazy

time_format = "%d-%m-%Y-%H-%M-%S"
time_pattern = "\d{1,2}-\d{1,2}-\d{4}-\d{1,2}-\d{1,2}-\d{1,2}"
folder_pattern = re.compile("model-(%s)" % time_pattern)
model_pattern = re.compile("model-(%s)\.ckpt\.meta" % time_pattern)


class Net(metaclass=ABCMeta):
    class NaNException(Exception):
        def __init__(self):
            super().__init__("NaN hasn't expected")

    @lazy.read_only_property
    def saver(self) -> tf.train.Saver:
        return tf.train.Saver(var_list=self.variables)

    @lazy.read_only_property
    def variables(self) -> Iterable[tf.Variable]:
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope)

    @property
    def create_time(self):
        if self._create_time is None:
            self._create_time = time.strftime(time_format)
        return self._create_time

    @property
    def folder_path(self) -> str:
        return "{}/model-{}".format(self.save_path, self.create_time)

    @property
    def model_path(self) -> str:
        return "{}/model-{}.ckpt".format(self.folder_path, time.strftime(time_format))

    def __init__(self, name: str, save_path: str, scope=None):
        self.save_path = save_path
        self.name = name
        self.scope = scope
        self._create_time = None

    def save(self, session: tf.Session):
        if not os.path.isdir(self.folder_path):
            os.mkdir(self.folder_path)
        self.saver.save(session, self.model_path)

    def newest(self, path: str, filtrator):
        names = [path + "/" + name for name in os.listdir(path) if filtrator(path, name)]
        if len(names) == 0:
            raise FileNotFoundError("Saves from {} net is not found".format(self.name))
        names.sort(key=os.path.getmtime, reverse=True)
        return names[0]

    def restore(self, session: tf.Session, model_path: str = None):
        folder_filtrator = lambda path, name: os.path.isdir(path + "/" + name) and re.match(folder_pattern, name)
        model_filtrator = lambda path, name: os.path.isfile(path + "/" + name) and re.match(model_pattern, name)
        if not model_path:
            folder_path = self.newest(self.save_path, folder_filtrator)
            model_path = self.newest(folder_path, model_filtrator)
            matched = re.match(model_pattern, model_path.split("/")[-1])
            self._create_time = matched.groups()[0]
            model_path = ".".join(model_path.split(".")[:-1])
        self.saver.restore(session, model_path)
