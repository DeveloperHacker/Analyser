import logging

from typing import List, Iterable


class Formatter:
    def __init__(self, heads: List[str], formats: List[str], sizes: List[int], rows: Iterable[int], height: int = None):
        self.vd = "│"
        self.bvd = "║"
        self.uvd = "┬"
        # noinspection SpellCheckingInspection
        self.lubvd = "╓"
        # noinspection SpellCheckingInspection
        self.rubvd = "╖"
        self.hd = "─"

        self._height = height
        self._row = 0
        head_segments = []
        line_segments = []
        delimiter_segments = []
        for i in rows:
            head_segments.append("{{segments[{}]:^{}s}}".format(i, sizes[i]))
            line_segments.append("{{segments[{}]:^{}{}}}".format(i, sizes[i], formats[i]))
            delimiter_segments.append(self.hd * sizes[i])
        self.head = self.bvd + self.vd.join(head_segments) + self.bvd
        self.line = self.bvd + self.vd.join(line_segments) + self.bvd
        self.delimiter = self.bvd + self.vd.join(delimiter_segments) + self.bvd
        self.upper_delimiter = self.lubvd + self.uvd.join(delimiter_segments) + self.rubvd
        self.head = self.head.format(segments=heads)

    def print(self, *args):
        if self._height and self._row % self._height == 0:
            self.print_head()
        logging.info(self.line.format(segments=list(args)))
        self._row += 1

    def print_head(self):
        if self._row:
            self.print_delimiter()
        else:
            self.print_upper_delimiter()
        self.print_head_text()
        self.print_delimiter()

    def print_delimiter(self):
        logging.info(self.delimiter)

    def print_upper_delimiter(self):
        logging.info(self.upper_delimiter)

    def print_head_text(self):
        logging.info(self.head)

    def print_appendix(self, text):
        for line in text.split("\n"):
            logging.info(self.bvd + " " + line)
        self._row = 0

    @property
    def height(self):
        return self._height

    @height.setter
    def height(self, height: int):
        self._height = height
        self._row = self._height or 0
