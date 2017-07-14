from config import init
from prepare_data_set import prepare_data_set
from seq2seq.AnalyserNet import AnalyserNet

if __name__ == '__main__':
    init()
    prepare_data_set()
    net = AnalyserNet()
    net.pretrain()
    net.train()
    net.test()
