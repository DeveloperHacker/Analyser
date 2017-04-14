import argparse
import logging
import sys

import word2vec.word2vec as word2vec
from seq2seq import contracts
from seq2seq.AnalyserNet import AnalyserNet
from seq2seq.MunchhausenNet import MunchhausenNet
from seq2seq.QFunctionNet import QFunctionNet
from variables.path import *


def main(args):
    if args.embedding:
        logging.basicConfig(level=logging.INFO, filename=WORD2VEC_LOG)
        logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
        word2vec.start(args.embedding)
    elif args.analyser:
        logging.basicConfig(level=logging.INFO, filename=ANALYSER_LOG)
        logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
        AnalyserNet.start(args.analyser)
    elif args.q_function:
        logging.basicConfig(level=logging.INFO, filename=Q_FUNCTION_LOG)
        logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
        QFunctionNet.start(args.q_function)
    elif args.contracts:
        logging.basicConfig(level=logging.INFO, filename=CONTRACTS_LOG)
        logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
        contracts.start(args.contracts)
    elif args.data_set:
        logging.basicConfig(level=logging.INFO, filename=GENERATOR_LOG)
        logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
        if "analyser" in args.data_set:
            AnalyserNet.build_data_set()
        elif "q_function" in args.data_set:
            QFunctionNet.build_data_set()
        elif "munchhausen" in args.data_set:
            MunchhausenNet.build_data_set()
    elif args.munchhausen:
        logging.basicConfig(level=logging.INFO, filename=MUNCHHAUSEN_LOG)
        logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
        MunchhausenNet.start(args.munchhausen)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-V", "--version", action='version', version='JavaDocs Analyser 0.0.1')
    parser.add_argument("--embedding", choices=["train", "cluster"], default=False)
    parser.add_argument("--analyser", choices=["train", "restore", "test"], default=False)
    parser.add_argument("--q_function", choices=["train", "restore", "test"], default=False)
    parser.add_argument("--contracts", choices=["train", "restore", "test"], default=False)
    parser.add_argument("--munchhausen", choices=["run", "train", "restore", "test"], default=False)
    parser.add_argument("--data_set", choices=["analyser", "q_function", "munchhausen"], default=False)
    main(parser.parse_args(sys.argv[1:]))
