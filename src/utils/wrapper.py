import logging
import time


def expand(clock: float):
    clock = int(clock * 1000)
    s, ms = divmod(clock, 1000)
    m, s = divmod(s, 60)
    h, m = divmod(m, 60)
    d, h = divmod(h, 24)
    if d != 0:
        return "{:d}d {:02d}h {:02d}m {:02d}s {:03d}ms".format(d, h, m, s, ms)
    elif h != 0:
        return "{:02d}h {:02d}m {:02d}s {:03d}ms".format(h, m, s, ms)
    elif m != 0:
        return "{:02d}m {:02d}s {:03d}ms".format(m, s, ms)
    elif s != 0:
        return "{:02d}s {:03d}ms".format(s, ms)
    else:
        return "{:03d}ms".format(ms)


def static(**kwargs):
    def decorate(func):
        for k in kwargs:
            setattr(func, k, kwargs[k])
        return func

    return decorate


@static(nested=0)
def trace(foo):
    def wrapper(*args, **kwargs):
        name = foo.__module__.split(".")[-1] + "." + foo.__name__
        shift = "│" * trace.nested
        start_time = time.time()
        expanded = expand(start_time)
        logging.info("{}╒Function \"{}\" is invoked in {}".format(shift, name, expanded))
        trace.nested += 1
        result = foo(*args, **kwargs)
        trace.nested -= 1
        end_time = time.time()
        delay = end_time - start_time
        expanded = expand(delay)
        logging.info("{}╘Function \"{}\" worked for {}".format(shift, name, expanded))
        return result

    return wrapper
