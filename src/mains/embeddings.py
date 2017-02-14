import _pickle as pickle

from mains.variables import *
from utils import batcher
from utils import filter
from utils import generator
from utils import unpacker

if __name__ == '__main__':
    methods = unpacker.unpackMethods(METHODS)
    docs = filter.applyFiltersForMethods(methods)
    with open(FILTERED, "w") as file:
        file.write("\n".join((text for doc in docs for label, text in doc)))
    generator.generateEmbeddings(DATA_SETS, EMB_MODEL, EMB_STORAGE, FILTERED, EPOCHS, FEATURES, WINDOW)
    embeddings = unpacker.unpackEmbeddings(EMB_STORAGE, EMB_MODEL)
    data = generator.vectorization(docs, embeddings)
    with open(VEC_METHODS, "wb") as file:
        pickle.dump(data, file)
    batches = batcher.batching(data, BATCH_SIZE, [MAX_ENCODE_SEQUENCE])
    with open(BATCHES, "wb") as file:
        pickle.dump(batches, file)
