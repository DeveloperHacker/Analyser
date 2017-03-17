import argparse
import logging
import sys

import word2vec.word2vec as word2vec
from seq2seq import contract, analyser, q_function
from utils import handlers
from variables.path import *


def main(args):
    handlers.sigint()
    if args.embedding:
        logging.basicConfig(level=logging.INFO, filename=WORD2VEC_LOG)
        logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
        word2vec.run(args.embedding)
    elif args.analyser:
        logging.basicConfig(level=logging.INFO, filename=ANALYSER_LOG)
        logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
        analyser.run(args.analyser)
    elif args.q_function:
        logging.basicConfig(level=logging.INFO, filename=Q_FUNCTION_LOG)
        logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
        q_function.run(args.q_function)
    elif args.contract:
        logging.basicConfig(level=logging.INFO, filename=SEQ2SEQ_LOG)
        logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
        contract.run(args.contract)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--version", action='version', version='JavaDocs Analyser 0.0.1')
    parser.add_argument("--embedding", nargs="?", choices=["train", "cluster"], const=True, default=False)
    parser.add_argument("--analyser", nargs="?", choices=["train", "restore", "test"], const=True,
                        default=False)
    parser.add_argument("--q_function", nargs="?", choices=["train", "restore", "test"], const=True,
                        default=False)
    parser.add_argument("--contract", nargs="?", choices=["train", "restore", "test"], const=True,
                        default=False)
    main(parser.parse_args(sys.argv[1:]))
