import logging
import os
import sys

from live_plotter.proxy.ProxyFigure import ProxyFigure

from constants.paths import ANALYSER_TRAIN_LOG
from seq2seq.AnalyserNet import AnalyserNet

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, filename=ANALYSER_TRAIN_LOG)
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    net = AnalyserNet()
    net.pretrain()
    net.train()
    net.test()
    ProxyFigure.destroy()
