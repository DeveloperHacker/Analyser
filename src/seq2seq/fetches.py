import tensorflow as tf

from utils.wrapper import trace
from variables.embeddings import TOKENS
from variables.train import *


def l2_loss():
    _VARIABLES = [var for var in tf.global_variables() if var.name in REGULARIZATION_VARIABLES]
    assert len(_VARIABLES) == len(REGULARIZATION_VARIABLES)
    return tf.reduce_sum([tf.nn.l2_loss(var) for var in _VARIABLES])


def distance(vector1, vector2):
    # ToDo: return tf.norm(vector1 - vector2)
    return tf.sqrt(tf.reduce_sum(tf.squared_difference(vector1, vector2), 1))


@trace
def pretrain(_OUTPUTS):
    embeddings = tf.stack(tuple(embedding for _, embedding in TOKENS))
    contains = lambda x: [tf.reduce_min(distance(vector, embeddings)) for vector in x]
    _CONTAINS_LOSS = tf.reduce_mean(tf.map_fn(contains, _OUTPUTS))
    _VARIANCE_LOSS = 1.0 / tf.reduce_mean(tf.square(tf.nn.moments(tf.stack(_OUTPUTS), [0])[1]))
    _L2_LOSS = l2_loss()
    _LOSS = tf.sqrt(
        tf.square(CONTAINS_WEIGHT * _CONTAINS_LOSS) +
        tf.square(VARIANCE_WEIGHT * _VARIANCE_LOSS) +
        tf.square(L2_WEIGHT * _L2_LOSS))
    _TRAIN = tf.train.AdamOptimizer(beta1=0.90).minimize(_LOSS)
    return _TRAIN, _LOSS, _CONTAINS_LOSS, _VARIANCE_LOSS, _L2_LOSS
