from utils import constructor
from utils import dumper
from utils import filter
from utils.wrapper import trace
from variables import *


@trace
def train():
    batches = dumper.load(BATCHES)[MAX_ENCODE_SEQUENCE]
    (fetches, _, _), feed_dicts = constructor.initRNN(batches, filter.parts)
    constructor.trainRNN(fetches, feed_dicts, False)


@trace
def restore():
    batches = dumper.load(BATCHES)[MAX_ENCODE_SEQUENCE]
    (fetches, _, _), feed_dicts = constructor.initRNN(batches, filter.parts)
    constructor.trainRNN(fetches, feed_dicts, True)


@trace
def test():
    embeddings = dumper.load(EMBEDDINGS)
    batches = dumper.load(BATCHES)[MAX_ENCODE_SEQUENCE]
    ((_, loss, _), (vars_BATCH, _, _), res_OUTPUTS), feed_dicts = constructor.initRNN(batches, filter.parts)
    constructor.testRNN(vars_BATCH, loss, res_OUTPUTS, feed_dicts, embeddings, filter.parts)
