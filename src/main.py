import _pickle as pickle
import argparse
import sys

from utils import batcher
from utils import constructor
from utils import filter
from utils import generator
from utils import printer
from utils import unpacker
from variables import *

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--version", action='version', version='JavaDocs Analyser 0.0.1')
    parser.add_argument("--embeddings", nargs="?", choices=["cluster"], const=True, default=False)
    parser.add_argument("--seq2seq", nargs="?", choices=["continue", "test"], const=True, default=False)
    parser.add_argument("--res", nargs="?", help="path to resources folder")
    args = parser.parse_args(sys.argv[1:])
    RESOURCES = args.res or RESOURCES
    if args.embeddings:
        if isinstance(args.embeddings, list):
            _, embeddings = unpacker.unpackEmbeddings()
            clusters = generator.generateClustersXNeighbors(embeddings, 0.3)
            printer.XNeighbors(clusters)

        else:
            methods = unpacker.unpackMethods(METHODS)
            docs = filter.applyFiltersForMethods(methods)
            with open(FILTERED, "w") as file:
                file.write("\n".join((text for doc in docs for label, text in doc)))
            generator.generateEmbeddings(DATA_SETS, EMB_MODEL, EMB_STORAGE, FILTERED, EMB_EPOCHS, EMB_SIZE, WINDOW)
            embeddings = unpacker.unpackEmbeddings()
            data = generator.vectorization(docs, embeddings)
            with open(VEC_METHODS, "wb") as file:
                pickle.dump(data, file)
            baskets = batcher.throwing(data, [MAX_ENCODE_SEQUENCE])
            batches = {basket: batcher.batching(data, BATCH_SIZE) for basket, data in baskets.items()}
            with open(BATCHES, "wb") as file:
                pickle.dump(batches, file)
    if args.seq2seq:
        if isinstance(args.seq2seq, list):
            with open(BATCHES, "rb") as file:
                batches = pickle.load(file)
            rnn = constructor.trainRNN(batches[MAX_ENCODE_SEQUENCE])
        else:
            pass
