import numpy as np
from live_plotter.proxy.ProxyFigure import ProxyFigure

from utils.wrappers import Timer


def run_multiply(length: int) -> float:
    timer = Timer("MULTIPLY")
    timer.start()
    max([2] * length)
    timer.stop()
    return timer.delay() * 1e3


def run_generate(length: int) -> float:
    timer = Timer("GENERATE")
    timer.start()
    max([2 for _ in range(length)])
    timer.stop()
    return timer.delay() * 1e3


def main(start, stop, experiments, cycles):
    with ProxyFigure(save_path="test.png") as figure:
        multiply_curve = figure.curve(1, 1, 1, "-r")
        generate_curve = figure.curve(1, 1, 1, "-b")
        figure.set_x_label(1, 1, 1, "cycles")
        figure.set_y_label(1, 1, 1, "t, ms")
        for length in np.linspace(start, stop, experiments):
            length = int(length)
            y_multiply = np.mean([run_multiply(length) for _ in range(cycles)])
            y_generate = np.mean([run_generate(length) for _ in range(cycles)])
            multiply_curve.append(length, y_multiply)
            generate_curve.append(length, y_generate)
            figure.draw()
            figure.save()


main(1, 10000, 100, 1000)
