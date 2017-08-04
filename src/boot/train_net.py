from configurations.paths import ANALYSER_DATA_SET
from seq2seq.AnalyserNet import AnalyserNet
from utils import dumpers

if __name__ == '__main__':
    data_set = dumpers.pkl_load(ANALYSER_DATA_SET)
    net = AnalyserNet(data_set)
    net.train()
