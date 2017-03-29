import argparse
import logging
import sys

import word2vec.word2vec as word2vec
from seq2seq import contracts, analyser, q_function
from seq2seq.analyser import AnalyserNet
from seq2seq.q_function import QFunctionNet
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
        AnalyserNet.run(args.analyser)
    elif args.q_function:
        logging.basicConfig(level=logging.INFO, filename=Q_FUNCTION_LOG)
        logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
        QFunctionNet.run(args.q_function)
    elif args.contracts:
        logging.basicConfig(level=logging.INFO, filename=CONTRACTS_LOG)
        logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
        contracts.run(args.contracts)
    elif args.data_set:
        logging.basicConfig(level=logging.INFO, filename=GENERATOR_LOG)
        logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
        if "analyser" in args.data_set:
            AnalyserNet.build_data_set()
        if "q_function" in args.data_set:
            QFunctionNet.build_data_set()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--version", action='version', version='JavaDocs Analyser 0.0.1')
    parser.add_argument("--embedding", choices=["train", "cluster"], default=False)
    parser.add_argument("--analyser", choices=["train", "restore", "test"], default=False)
    parser.add_argument("--q_function", choices=["train", "restore", "test"], default=False)
    parser.add_argument("--contracts", choices=["train", "restore", "test"], default=False)
    parser.add_argument("--data_set", choices=["analyser", "q_function"], default=False)
    main(parser.parse_args(sys.argv[1:]))
