import collections
import time

from logger import logger


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


def optional_arg_decorator(func):
    def wrapped_decorator(*args):
        if len(args) == 1 and callable(args[0]):
            return func(args[0])
        else:
            def real_decorator(decorate):
                return func(decorate, *args)

            return real_decorator

    return wrapped_decorator


def static(**kwargs):
    def decorate(func):
        for k in kwargs:
            setattr(func, k, kwargs[k])
        return func

    return decorate


@optional_arg_decorator
def trace(func, name=None):
    def wrapped(*args, **kwargs):
        with Timer(name or func.__name__):
            result = func(*args, **kwargs)
        return result

    return wrapped


class Timer:
    def __init__(self, name: str = None, printer=logger.info):
        self.begin = 0
        self.end = 0
        self.name = name
        self.printer = printer

    def start(self):
        self.begin = time.time()

    def stop(self):
        self.end = time.time()

    def delay(self):
        return self.end - self.begin

    def __enter__(self):
        self.start()
        if self.printer is not None:
            self.printer("process '{}' is started".format(self.name))
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
        if self.printer is not None:
            expanded = expand_time(self.delay())
            formatted = format_time(*expanded)
            formatter = "process '{}' worked for '{}'"
            logger.info(formatter.format(self.name, formatted))


class memoize:
    @staticmethod
    def read_only_property(func):
        attr_name = "_lazy_" + func.__name__

        @property
        def _property(self):
            if not hasattr(self, attr_name):
                setattr(self, attr_name, func(self))
            return getattr(self, attr_name)

        @_property.setter
        def _property(*_):
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
        attr_name = "_cache_" + func.__name__

        def _method(self, *args):
            if not isinstance(args, collections.Hashable):
                return func(self, *args)
            if not hasattr(self, attr_name):
                setattr(self, attr_name, {})
            cache = getattr(self, attr_name)
            if args not in cache:
                value = func(self, *args)
                cache[args] = value
            return cache[args]

        return _method

    @staticmethod
    def function(func):
        cache = {}

        def _function(*args):
            if not isinstance(args, collections.Hashable):
                return func(*args)
            if args not in cache:
                value = func(*args)
                cache[args] = value
            return cache[args]

        return _function
