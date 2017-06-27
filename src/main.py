from config import init
from prepare_data_set import prepare_data_set
from train_net import train_net
from utils.wrapper import trace


@trace
def main():
    prepare_data_set()
    net = train_net()
    net.test()


if __name__ == '__main__':
    init()
    main()
