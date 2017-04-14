import logging


class MunchhausenFormatter:
    def __init__(self, tracers: list, losses: list, row_size: int = 11, column_size: int = 10):
        self.row_size = row_size
        self.column_size = column_size
        f1 = "{{:^{size:d}s}}"
        f2 = "{{:^{size:d}d}}"
        f3 = "{{:^{size:d}.4f}}"
        f4 = "│".join((f1,) * len(tracers))
        f5 = "│".join((f1,) * len(losses))
        f6 = "│".join((f3,) * len(losses))
        f7 = "─" * row_size
        f8 = "│".join((f7,) * len(tracers))
        f9 = "│".join((f7,) * len(losses))
        self.head = ("║" + f4 + "║" + f5 + "║" + f5 + "║").format(size=row_size).format(*tracers, *(losses * 2))
        self.line = ("║" + f2 + "│" + f3 + "║" + f6 + "║" + f6 + "║").format(size=row_size)
        self.delimiter = ("╟" + f8 + "╫" + f9 + "╫" + f9 + "╢")
        self.row = 0

    def run(self, model_name: str):
        self.row = 0
        logging.info("RUN model: {}".format(model_name))

    def print(self, *args):
        if self.row % self.column_size == 0:
            if self.row > 0:
                logging.info(self.delimiter)
            logging.info(self.head)
            logging.info(self.delimiter)
        logging.info(self.line.format(*args))
        self.row += 1
