from utils import constructor
from variables import BATCHES
import _pickle as pickle
import numpy as np

if __name__ == '__main__':
    with open(BATCHES, "rb") as file:
        batches = pickle.load(file)
    # print(len(batches))
    # print(np.mean([np.mean(np.std([constructor.vectorization(doc) for doc in batch], 0)) for batch in batches]))
    rnn = constructor.constructRNNNet(batches)
