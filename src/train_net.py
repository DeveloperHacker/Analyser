from config import init
from seq2seq.AnalyserNet import AnalyserNet

if __name__ == '__main__':
    init()
    net = AnalyserNet()
    net.pretrain()
    net.train()
