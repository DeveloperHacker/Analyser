import logging
import sys

from seq2seq.AnalyserNet import AnalyserNet
from constants.paths import ANALYSER_TEST_LOG

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, filename=ANALYSER_TEST_LOG)
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    net = AnalyserNet()
    net.test()
