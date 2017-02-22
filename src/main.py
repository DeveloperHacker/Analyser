import argparse
import logging
import sys

import embeddings
import seq2seq
from variables import RESOURCES

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--version", action='version', version='JavaDocs Analyser 0.0.1')
    parser.add_argument("--embeddings", nargs="?", choices=["train", "cluster"], const=True, default=False)
    parser.add_argument("--seq2seq", nargs="?", choices=["train", "continue", "test"], const=True, default=False)
    parser.add_argument("--res", nargs="?", help="path to resources folder")
    args = parser.parse_args(sys.argv[1:])
    RESOURCES = args.res or RESOURCES

    logging.basicConfig(level=logging.INFO)
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    if args.embeddings:
        if "train" == args.embeddings:
            embeddings.generate()
        elif "cluster" == args.embeddings:
            embeddings.cluster()
    elif args.seq2seq:
        if "train" == args.seq2seq:
            seq2seq.train()
        elif "continue" == args.seq2seq:
            seq2seq.restore()
        elif "test" == args.seq2seq:
            seq2seq.test()
