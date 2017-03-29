import numpy as np

from utils import dumper as _dumper
from variables import path as _path
from variables.tags import GO, PAD
from variables.train import *
import tensorflow as tf


class Embeddings:
    _instance = None
    _word2idx = None
    _emb2idx = None

    @staticmethod
    def instance() -> list:
        if Embeddings._instance is None:
            Embeddings._instance = _dumper.load(_path.EMBEDDINGS)
            Embeddings._instance[GO] = np.zeros([EMBEDDING_SIZE], dtype=np.float32)
            Embeddings._instance[PAD] = np.ones([EMBEDDING_SIZE], dtype=np.float32)
            Embeddings._instance = list(Embeddings._instance.items())
            Embeddings._instance.sort(key=lambda x: x[0])
        # noinspection PyTypeChecker
        return Embeddings._instance

    @staticmethod
    def words():
        return [word for word, embedding in Embeddings.instance()]

    @staticmethod
    def embeddings():
        return [embedding for word, embedding in Embeddings.instance()]

    @staticmethod
    def emb2idx() -> dict:
        if Embeddings._emb2idx is None:
            Embeddings._emb2idx = {}
            for index, (word, embedding) in enumerate(Embeddings.instance()):
                Embeddings._emb2idx[str(embedding)] = index
        # noinspection PyTypeChecker
        return Embeddings._emb2idx

    @staticmethod
    def word2idx() -> dict:
        if Embeddings._word2idx is None:
            Embeddings._word2idx = {}
            for index, (word, embedding) in enumerate(Embeddings.instance()):
                Embeddings._word2idx[word] = index
        # noinspection PyTypeChecker
        return Embeddings._word2idx

    @staticmethod
    def get_store(key):
        if isinstance(key, int):
            index = key
        elif isinstance(key, str):
            if key in Embeddings.word2idx():
                index = Embeddings.word2idx()[key]
            else:
                index = Embeddings.word2idx()["UNK"]
        else:
            key = str(key)
            if key in Embeddings.emb2idx():
                index = Embeddings.emb2idx()[key]
            else:
                raise Exception("Store with embedding {} is not found".format(key))
        return (index,) + Embeddings.instance()[index]

    @staticmethod
    def get_embedding(key):
        return Embeddings.get_store(key)[2]

    @staticmethod
    def get_index(key):
        return Embeddings.get_store(key)[0]

    @staticmethod
    def get_word(key):
        return Embeddings.get_store(key)[1]


INITIAL_STATE = tf.zeros([BATCH_SIZE, INPUT_STATE_SIZE], dtype=np.float32)
