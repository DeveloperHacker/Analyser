from config import init
from seq2seq.AnalyserNet import AnalyserNet
from utils.wrapper import trace


@trace
def test_net():
    net = AnalyserNet()
    net.test()


if __name__ == '__main__':
    init()
    test_net()
