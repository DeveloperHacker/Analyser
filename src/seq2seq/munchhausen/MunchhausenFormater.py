import logging

from typing import List, Iterable


class MunchhausenFormatter:
    def __init__(self, heads: List[str], formats: List[str], sizes: List[int], rows: Iterable[int], height: int):
        self.height = height
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
        self.row = 0

    def run(self, model_name: str):
        self.row = 0
        print("RUN model: {}".format(model_name))

    def print(self, *args):
        if self.row % self.height == 0:
            if self.row > 0:
                logging.info(self.delimiter)
            logging.info(self.head)
            logging.info(self.delimiter)
        logging.info(self.line.format(segments=list(args)))
        self.row += 1
