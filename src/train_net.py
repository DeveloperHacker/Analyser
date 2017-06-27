import logging

from live_plotter.proxy.ProxyFigure import ProxyFigure

from config import init
from seq2seq.AnalyserNet import AnalyserNet
from seq2seq.Net import Net
from utils.wrapper import trace


@trace
def train_net():
    try:
        net = AnalyserNet()
        net.pretrain()
        try:
            net.train()
        except Net.NaNException as ex:
            logging.info(ex)
    finally:
        ProxyFigure.destroy()
    return net


if __name__ == '__main__':
    init()
    train_net()
