import logging
import time


def expand_time(clock: float):
    clock = int(clock * 1000)
    s, ms = divmod(clock, 1000)
    m, s = divmod(s, 60)
    h, m = divmod(m, 60)
    d, h = divmod(h, 24)
    return d, h, m, s, ms


def format_time(d: int, h: int, m: int, s: int, ms: int):
    if d != 0:
        return "{:d}days {:02d}:{:02d}:{:02d}.{:03d}".format(d, h, m, s, ms)
    elif h != 0:
        return "{:02d}:{:02d}:{:02d}.{:03d}".format(h, m, s, ms)
    elif m != 0:
        return "{:02d}:{:02d}.{:03d}".format(m, s, ms)
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
def trace(func):
    def wrapper(*args, **kwargs):
        name = func.__module__.split(".")[-1] + "." + func.__name__
        shift = "│" * trace.nested
        start_time = time.time()
        d, h, m, s, ms = expand_time(start_time)
        formatted = format_time(0, (h + 3) % 24, m, s, ms)
        logging.info("{}╒Function \"{}\" is invoked at {} o'clock".format(shift, name, formatted))
        trace.nested += 1
        try:
            result = func(*args, **kwargs)
        finally:
            trace.nested -= 1
            end_time = time.time()
            delay = end_time - start_time
            expanded = expand_time(delay)
            formatted = format_time(*expanded)
            logging.info("{}╘Function \"{}\" worked for {}".format(shift, name, formatted))
        return result

    return wrapper


def read_only_lazy_property(func):
    attr_name = "_lazy_" + func.__name__

    @property
    def _property(self):
        if not hasattr(self, attr_name):
            setattr(self, attr_name, func(self))
        return getattr(self, attr_name)

    @_property.setter
    def _property(self, value):
        raise AttributeError

    return _property


def lazy_property(func):
    attr_name = "_lazy_" + func.__name__

    @property
    def _property(self):
        if not hasattr(self, attr_name):
            setattr(self, attr_name, func(self))
        return getattr(self, attr_name)

    @_property.setter
    def _property(self, value):
        setattr(self, attr_name, value)

    return _property


def lazy_method(func):
    attr_name = "_lazy_" + func.__name__

    def _method(self, *args, **kwargs):
        if not hasattr(self, attr_name):
            setattr(self, attr_name, func(self, *args, **kwargs))
        return getattr(self, attr_name)

    return _method


def lazy_function(func):
    instance = None

    def _function(*args, **kwargs):
        nonlocal instance
        if instance is None:
            instance = func(*args, **kwargs)
        return instance

    return _function
