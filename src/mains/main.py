import _pickle as pickle

from mains.variables import BATCHES, BATCH_SIZE
from utils import constructor

if __name__ == '__main__':
    with open(BATCHES, "rb") as file:
        batches = pickle.load(file)
    rnn = constructor.constructRNNNet(batches, BATCH_SIZE)
