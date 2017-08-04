from boot.prepare_data_set import prepare_data_set
from seq2seq.AnalyserNet import AnalyserNet

if __name__ == '__main__':
    data_set = prepare_data_set(dump=False)
    net = AnalyserNet(data_set)
    net.train()
    net.test()
