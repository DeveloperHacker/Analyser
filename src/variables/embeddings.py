import numpy as np
import tensorflow as tf

from utils import dumper as _dumper
from variables import path as _path
from variables.train import *

EMBEDDINGS = _dumper.load(_path.EMBEDDINGS)

GO = np.zeros([EMBEDDING_SIZE], dtype=np.float32)
PAD = np.ones([EMBEDDING_SIZE], dtype=np.float32)
INITIAL_STATE = tf.zeros([BATCH_SIZE, INPUT_STATE_SIZE], dtype=np.float32)
