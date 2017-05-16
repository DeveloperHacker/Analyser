import argparse
import logging
import sys

from seq2seq.munchhausen.MunchhausenNet import MunchhausenNet
from variables.paths import WORD2VEC_LOG, MUNCHHAUSEN_LOG
from word2vec.word2vec_optimized import word2vec


def main(args):
    if args.embedding:
        logging.basicConfig(level=logging.INFO, filename=WORD2VEC_LOG)
        logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
        word2vec.start(args.embedding)
    elif args.munchhausen:
        logging.basicConfig(level=logging.INFO, filename=MUNCHHAUSEN_LOG)
        logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
        MunchhausenNet.start(args.munchhausen)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-V", "--version", action='version', version='JavaDocs Analyser 0.0.1')
    parser.add_argument("--embedding", choices=["train", "cluster"], default=False)
    parser.add_argument("--munchhausen", choices=["run", "train", "restore", "test", "data_set"], default=False)
    main(parser.parse_args(sys.argv[1:]))
