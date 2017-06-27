import logging

from typing import Iterable, Any


def chunks(iterable: Iterable[Any], block_size: int):
    result = []
    for element in iterable:
        result.append(element)
        if len(result) == block_size:
            yield result
            result = []
    if len(result) > 0:
        yield result


class Formatter:
    VD = "│"
    BVD = "║"
    UVD = "┬"
    LUBVD = "╓"
    RUBVD = "╖"
    LVD = "┴"
    LLBVD = "╙"
    RLBVD = "╜"
    HD = "─"

    def __init__(self,
                 heads: Iterable[str],
                 formats: Iterable[str],
                 sizes: Iterable[int],
                 rows: Iterable[int],
                 height: int = None):
        self._row = 0
        self._height = height
        self._heads = list(heads)
        self._formats = list(formats)
        self._sizes = list(sizes)
        self._rows = list(rows)
        head_segments = []
        line_segments = []
        delimiter_segments = []
        for row in self._rows:
            head_segments.append("{{segments[{}]:^{}s}}".format(row, self._sizes[row]))
            line_segments.append("{{segments[{}]:^{}{}}}".format(row, self._sizes[row], self._formats[row]))
            delimiter_segments.append(self.HD * self._sizes[row])
        self._head = self.BVD + self.VD.join(head_segments) + self.BVD
        self._line = self.BVD + self.VD.join(line_segments) + self.BVD
        self._delimiter = self.BVD + self.VD.join(delimiter_segments) + self.BVD
        self._upper_delimiter = self.LUBVD + self.UVD.join(delimiter_segments) + self.RUBVD
        self._lower_delimiter = self.LLBVD + self.LVD.join(delimiter_segments) + self.RLBVD
        self._head = self._head.format(segments=self._heads)

    @staticmethod
    def _print(text: str):
        logging.info(text)

    def print(self, *args):
        if self._height and self._row % self._height == 0:
            self.print_head()
        self._print(self._line.format(segments=list(args)))
        self._row += 1

    def print_head(self):
        if self._row:
            self.print_delimiter()
        else:
            self.print_upper_delimiter()
        self.print_head_text()
        self.print_delimiter()

    def print_delimiter(self):
        self._print(self._delimiter)

    def print_upper_delimiter(self):
        self._print(self._upper_delimiter)
        self._row = 0

    def print_lower_delimiter(self):
        self._print(self._lower_delimiter)
        self._row = 0

    def print_head_text(self):
        self._print(self._head)

    def print_appendix(self, text: str, prefix: str = None):
        for line in text.split("\n"):
            size = sum(self._sizes[row] for row in self._rows) + len(self._rows) + 1 - 4
            if prefix is not None:
                left_size = self._sizes[self._rows[0]] if len(self._rows) > 0 else 0
                left_size = max(0, left_size - 2)
                right_size = size - left_size - 3
                for _line in chunks(line, right_size):
                    _line = "".join(_line).strip()
                    formatter = "{{}} {{:{}s}} {{}} {{:{}s}} {{}}".format(left_size, right_size)
                    _line = formatter.format(self.BVD, prefix, self.VD, _line, self.BVD)
                    self._print(_line)
                    prefix = ""
            else:
                for _line in chunks(line, size):
                    _line = "".join(_line).strip()
                    _line = "{{}} {{:{}s}} {{}}".format(size).format(self.BVD, _line, self.BVD)
                    self._print(_line)

    @property
    def height(self):
        return self._height

    @height.setter
    def height(self, height: int):
        self._height = height
        self._row = self._height or 0
