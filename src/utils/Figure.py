from matplotlib import pyplot


class Figure:
    def __init__(self, num=None, xmin=0, xmax=1, xauto=False, ymin=0, ymax=1, yauto=True, xlabel=None, ylabel=None,
                 resolution=None, grid=True, warn=False):
        if resolution is None: resolution = [6, 6]
        self._figure = pyplot.figure(num)
        self._figure.set_size_inches(*resolution)
        plot = self._figure.gca()
        plot.set_xlim(xmin=xmin, xmax=xmax, auto=xauto)
        plot.set_ylim(ymin=ymin, ymax=ymax, auto=yauto)
        plot.set_xlabel("" if xlabel is None else xlabel)
        plot.set_ylabel("" if ylabel is None else ylabel)
        plot.grid(grid)
        self._figure.show(warn)
        self._closed = False

    def plot(self, x, y, mode='.'):
        if self._closed:
            raise Exception("Figure {} is closed".format(self._figure.number))
        self._figure.gca().plot(x, y, mode)
        self._figure.canvas.draw()

    def save(self, filename=None):
        if filename is None:
            filename = "figure_{}.png".format(self._figure.number)
        self._figure.savefig(filename)

    def close(self):
        pyplot.close(self._figure)

    def show(self, warn=False):
        self._figure.show(warn)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def line(self, mode=".b") -> 'Figure.Line':
        return Figure.Line(self, mode)

    class Line:
        def __init__(self, parent: 'Figure', mode):
            self.axes = []
            self.ayes = []
            self.parent = parent
            self.mode = mode

        def append(self, x, y):
            self.axes.append(x)
            self.ayes.append(y)

        def plot(self, x, y):
            self.append(x, y)
            if len(self.axes) > 1:
                self.parent.plot(self.axes[-2:-1], self.ayes[-2:-1], self.mode)
            else:
                self.parent.plot(x, y, self.mode)
