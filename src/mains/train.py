import _pickle as pickle

from mains.variables import *
from utils import constructor

if __name__ == '__main__':
    with open(BATCHES, "rb") as file:
        batches = pickle.load(file)[MAX_ENCODE_SEQUENCE]
    rnn = constructor.buildRNN(batches)
