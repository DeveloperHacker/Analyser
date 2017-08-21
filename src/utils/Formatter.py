from typing import Iterable

from logger import logger


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
                 rows: Iterable[int] = None,
                 height: int = None):
        self._row = 0
        self._height = height
        self._heads = list(heads)
        self._formats = list(formats)
        self._sizes = list(sizes)
        self._rows = list(rows or range(len(self._sizes)))
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
        # self._delimiter = self.BVD + self.HD.join(delimiter_segments) + self.BVD
        # self._upper_delimiter = self.LUBVD + self.HD.join(delimiter_segments) + self.RUBVD
        # self._lower_delimiter = self.LLBVD + self.HD.join(delimiter_segments) + self.RLBVD
        self._head = self._head.format(segments=self._heads)
        self.raw_print = logger.info

    def print(self, *args):
        if self._height and self._row % self._height == 0:
            if self._row > 0:
                self.print_delimiter()
            else:
                self.print_upper_delimiter()
            self.print_head_text()
            self.print_delimiter()
        self.raw_print(self._line.format(segments=list(args)))
        self._row += 1

    def print_head(self):
        self.print_upper_delimiter()
        self.print_head_text()
        self.print_delimiter()

    def print_delimiter(self):
        self.raw_print(self._delimiter)

    def print_upper_delimiter(self):
        self.raw_print(self._upper_delimiter)
        self._row = 0

    def print_lower_delimiter(self):
        self.raw_print(self._lower_delimiter)
        self._row = 0

    def print_head_text(self):
        self.raw_print(self._head)

    @property
    def height(self):
        return self._height

    @height.setter
    def height(self, height: int):
        self._height = height
        self._row = self._height or 0

    @property
    def size(self):
        return sum(self._sizes[row] for row in self._rows) + len(self._rows) + 1

    def row_size(self, row_number):
        return self._sizes[row_number]
