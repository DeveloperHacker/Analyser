import time
from queue import Queue
from threading import Thread

from matplotlib import pyplot


class Printer:
    _instance = None

    def __new__(cls, pause=1) -> 'Printer':
        return object.__new__(Printer) if Printer._instance is None else Printer._instance

    def __init__(self, pause=1):
        if Printer._instance is not None: return
        Printer._instance = self
        self._pause = pause
        self._plots = Queue()
        self._closes = Queue()
        self._current_figure = None
        self._thread = Thread(target=Printer._update, args=(self,))
        self._thread_init_stop = False
        self._thread_stop = True
        self._figures = {}

    def figure(self, uid=None):
        self._current_figure = uid

    def close(self, uid: int):
        self._closes.put(uid)

    def _update(self):
        self._thread_init_stop = False
        while not self._thread_init_stop or not self._plots.empty() or not self._closes.empty():
            while not self._plots.empty():
                uid, args = self._plots.get()
                figure = pyplot.figure(uid)
                if uid not in self._figures:
                    self._figures[uid] = figure.add_subplot(111)
                    figure.show(False)
                    figure.hold(True)
                self._figures[uid].plot(*args)
                figure.canvas.draw()
            while not self._closes.empty():
                uid = self._closes.get()
                pyplot.close(uid)
            time.sleep(0.1)
        self._thread_stop = True

    def plot(self, x, y, mode="."):
        self._plots.put((self._current_figure, (x, y, mode)))
        if self._thread_init_stop:
            raise Exception("Printer is killed")
        if self._thread_stop:
            self._thread_stop = False
            self._thread.start()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.kill()

    def kill(self):
        self._thread_init_stop = True
