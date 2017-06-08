import logging

from typing import List, Iterable


class Formatter:
    def __init__(self, heads: List[str], formats: List[str], sizes: List[int], rows: Iterable[int], height: int):
        self._height = height
        head_segments = []
        line_segments = []
        delimiter_segments = []
        for i in rows:
            head_segments.append("{{segments[{}]:^{}s}}".format(i, sizes[i]))
            line_segments.append("{{segments[{}]:^{}{}}}".format(i, sizes[i], formats[i]))
            delimiter_segments.append("─" * sizes[i])
        self.head = "║" + "│".join(head_segments) + "║"
        self.line = "║" + "│".join(line_segments) + "║"
        self.delimiter = "║" + "│".join(delimiter_segments) + "║"
        self.head = self.head.format(segments=heads)
        self._row = 0

    def run(self, model_name: str):
        self._row = 0
        print("RUN model: {}".format(model_name))

    def print(self, *args):
        if self._row % self._height == 0:
            if self._row > 0:
                logging.info(self.delimiter)
            logging.info(self.head)
            logging.info(self.delimiter)
        logging.info(self.line.format(segments=list(args)))
        self._row += 1

    @property
    def height(self):
        return self.height

    @height.setter
    def height(self, height: int):
        self._height = height
        self._row = self._height
