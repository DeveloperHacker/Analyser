import time

from configurations.logger import timing_logger


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


def trace(func):
    def wrapper(*args, **kwargs):
        with Timer(func.__name__):
            result = func(*args, **kwargs)
        return result

    return wrapper


class Timer:
    def __init__(self, name: str):
        self.begin = 0
        self.end = 0
        self.name = name

    def start(self):
        self.begin = time.time()

    def stop(self):
        self.end = time.time()

    def delay(self):
        return self.end - self.begin

    @staticmethod
    def log(text):
        timing_logger.info(text)

    def __enter__(self):
        self.start()
        self.log("process '%s' is started" % self.name)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
        expanded = expand_time(self.delay())
        formatted = format_time(*expanded)
        formatter = "process '{}' worked for '{}'"
        self.log(formatter.format(self.name, formatted))


class lazy:
    @staticmethod
    def read_only_property(func):
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

    @staticmethod
    def property(func):
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

    @staticmethod
    def method(func):
        attr_name = "_lazy_" + func.__name__

        def _method(self, *args, **kwargs):
            if not hasattr(self, attr_name):
                setattr(self, attr_name, func(self, *args, **kwargs))
            return getattr(self, attr_name)

        return _method

    @staticmethod
    def function(func):
        instance = None

        def _function(*args, **kwargs):
            nonlocal instance
            if instance is None:
                instance = func(*args, **kwargs)
            return instance

        return _function
