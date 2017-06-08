import logging
import sys

from live_plotter.proxy.ProxyFigure import ProxyFigure

from constants.paths import ANALYSER_TRAIN_LOG
from seq2seq.AnalyserNet import AnalyserNet
from seq2seq.Net import Net

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, filename=ANALYSER_TRAIN_LOG)
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    try:
        net = AnalyserNet()
        net.pretrain()
        try:
            net.train()
        except Net.NaNException as ex:
            print(ex)
        net.test()
    finally:
        ProxyFigure.destroy()
