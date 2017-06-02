import logging
import sys

from live_plotter.proxy.ProxyFigure import ProxyFigure

from seq2seq.AnalyserNet import AnalyserNet
from variables.paths import ANALYSER_TRAIN_LOG

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, filename=ANALYSER_TRAIN_LOG)
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    net = AnalyserNet()
    net.pretrain()
    net.train()
    net.test()
    ProxyFigure.destroy()
