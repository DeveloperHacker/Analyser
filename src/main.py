from prepare_data_set import prepare_data_set
from seq2seq.AnalyserNet import AnalyserNet
from utils.wrappers import Timer

if __name__ == '__main__':
    prepare_data_set()
    with Timer("BUILD"):
        net = AnalyserNet()
    with Timer("PRETRAIN"):
        net.pretrain()
    with Timer("TRAIN"):
        net.train()
    with Timer("TEST"):
        net.test()
