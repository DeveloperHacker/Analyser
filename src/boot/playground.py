import numpy as np
from live_plotter.proxy.ProxyFigure import ProxyFigure

from utils.wrappers import Timer


def run_map(cycles: int) -> float:
    xs = range(cycles)
    timer = Timer("MAP")
    timer.start()
    max(map(lambda x: x + 2, xs))
    timer.stop()
    return timer.delay() * 1e3


def run_list(cycles: int) -> float:
    xs = range(cycles)
    timer = Timer("LIST")
    timer.start()
    max([x + 2 for x in xs])
    timer.stop()
    return timer.delay() * 1e3


def main(start, stop, number, test_cycles):
    with ProxyFigure(save_path="test.png") as figure:
        map_curve = figure.curve(1, 1, 1, "-r")
        list_curve = figure.curve(1, 1, 1, "-b")
        figure.set_x_label(1, 1, 1, "cycles")
        figure.set_y_label(1, 1, 1, "t, ms")
        for cycles in np.linspace(start, stop, number):
            cycles = int(cycles)
            y_map = np.mean([run_map(cycles) for _ in range(test_cycles)])
            y_list = np.mean([run_list(cycles) for _ in range(test_cycles)])
            map_curve.append(cycles, y_map)
            list_curve.append(cycles, y_list)
            figure.draw()
            figure.save()


main(1, 10000, 100, 1000)
