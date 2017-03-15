import signal


class SIGINTException(Exception):
    pass


def sigint():
    def handler(signum, frame):
        raise SIGINTException()

    signal.signal(signal.SIGINT, handler)
