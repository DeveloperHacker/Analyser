import os
import re
import time
from abc import ABCMeta

import tensorflow as tf


class Net(metaclass=ABCMeta):
    class NaNException(Exception):
        def __init__(self):
            super().__init__("NaN hasn't expected")

    def get_saver(self) -> tf.train.Saver:
        if self._saver is None:
            self._saver = tf.train.Saver(var_list=self.get_variables())
        return self._saver

    def get_variables(self):
        if self._variables is None:
            self._variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope)
        return self._variables

    def __init__(self, name: str, save_path: str):
        self.save_path = save_path
        self.name = name
        self.create_time = time.strftime("%d-%m-%Y-%H-%M-%S")
        self._variables = None
        self._saver = None
        self.scope = None

    def reset(self, session: tf.Session):
        self.create_time = time.strftime("%d-%m-%Y-%H-%M-%S")
        self._saver = None
        session.run(tf.global_variables_initializer())

    def save(self, session: tf.Session):
        path = self.get_model_path()
        model_path = path + "/model-{}.ckpt".format(time.strftime("%d-%m-%Y-%H-%M-%S"))
        saver = self.get_saver()
        saver.save(session, model_path)

    def newest(self, path: str, filtrator):
        names = [path + "/" + name for name in os.listdir(path) if filtrator(path + "/" + name)]
        if len(names) == 0:
            raise FileNotFoundError("Saves from {} net is not found".format(self.name))
        names.sort(key=os.path.getmtime, reverse=True)
        last = names[0]
        return last

    def restore(self, session: tf.Session):
        filtrator = lambda path: os.path.isdir(path) and len(os.listdir(path)) > 0 and self.name in path
        folder_path = self.newest(self.save_path, filtrator)
        model_filtrator = lambda path: os.path.isfile(path) and re.match(r".+/model-.+\.ckpt\.meta", path)
        model_path = self.newest(folder_path, model_filtrator)
        model_path = ".".join(model_path.split(".")[:-1])
        self.create_time = re.match(".+/model-(.+)\.ckpt", model_path).groups()[0]
        self.get_saver().restore(session, model_path)

    def get_model_path(self) -> str:
        path = "{}/model-{}-net-{}".format(self.save_path, self.name, self.create_time)
        if not os.path.isdir(path): os.mkdir(path)
        return path
