import argparse
import logging
import sys

from seq2seq import analyser, q_function
import word2vec.word2vec as word2vec
from variables.path import *

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--version", action='version', version='JavaDocs Analyser 0.0.1')
    parser.add_argument("--word2vec", nargs="?", choices=["train", "cluster"], const=True, default=False)
    parser.add_argument("--analyser", nargs="?", choices=["train", "restore", "test"], const=True,
                        default=False)
    parser.add_argument("--q_function", nargs="?", choices=["train", "restore", "test"], const=True,
                        default=False)
    args = parser.parse_args(sys.argv[1:])

    # with tf.Session() as session:
    #     summary_writer = tf.summary.FileWriter(PATH, session.graph)
    #     session.run(tf.global_variables_initializer())
    #     summary_writer.close()

    if args.word2vec:
        logging.basicConfig(level=logging.INFO, filename=WORD2VEC_LOG)
        logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
        if "train" == args.word2vec:
            word2vec.train()
        elif "cluster" == args.word2vec:
            word2vec.cluster()
    elif args.analyser:
        logging.basicConfig(level=logging.INFO, filename=ANALYSER_LOG)
        logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
        if "train" == args.analyser:
            analyser.train()
        elif "restore" == args.analyser:
            analyser.train(True)
        elif "test" == args.analyser:
            analyser.test()
    elif args.q_function:
        logging.basicConfig(level=logging.INFO, filename=Q_FUNCTION_LOG)
        logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
        if "train" == args.q_function:
            q_function.train()
        elif "restore" == args.q_function:
            q_function.train(True)
        elif "test" == args.q_function:
            q_function.test()
