import logging
import time

nested = 0


def trace(f):
    def wrapper(*args, **kwargs):
        global nested
        name = f.__module__ + "." + f.__name__
        shift = "│" * nested
        logging.info("{}╒Function \"{}\" is invoked".format(shift, name))
        clock = time.time()
        nested += 1
        result = f(*args, **kwargs)
        nested -= 1
        delay = int((time.time() - clock) * 1000)
        ms = delay % 1000
        s = delay // 1000 % 60
        m = delay // 1000 // 60 % 60
        h = delay // 1000 // 60 // 60
        logging.info("{}╘Function \"{}\" worked for {:02d}h {:02d}m {:02d}s {:03d}ms".format(shift, name, h, m, s, ms))
        return result

    return wrapper
