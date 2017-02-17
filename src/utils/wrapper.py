import signal

class SIGINTException(Exception): pass


def sigint(f):
    def wrapper(*args, **kwargs):
        def handler(signum, frame):
            raise SIGINTException()

        signal.signal(signal.SIGINT, handler)
        return f(*args, **kwargs)
    return wrapper
