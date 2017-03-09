import argparse
import logging
import sys

import seq2seq.seq2seq as seq2seq
import word2vec.word2vec as word2vec
from variables.path import *

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--version", action='version', version='JavaDocs Analyser 0.0.1')
    parser.add_argument("--word2vec", nargs="?", choices=["train", "cluster"], const=True, default=False)
    parser.add_argument("--seq2seq", nargs="?", choices=["train", "restore", "test"], const=True, default=False)
    args = parser.parse_args(sys.argv[1:])

    if args.word2vec:
        logging.basicConfig(level=logging.INFO, filename=WORD2VEC_LOG)
        logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
        if "train" == args.word2vec:
            word2vec.train()
        elif "cluster" == args.word2vec:
            word2vec.cluster()
    elif args.seq2seq:
        logging.basicConfig(level=logging.INFO, filename=SEQ2SEQ_LOG)
        logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
        if "train" == args.seq2seq:
            seq2seq.train()
        elif "restore" == args.seq2seq:
            seq2seq.train(True)
        elif "test" == args.seq2seq:
            seq2seq.test()
