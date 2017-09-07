import matplotlib.pyplot as plt
import numpy as np

from analyser.Score import Score
from analyser.misc import print_scores
from analyser_main import sub
from utils import dumpers


def print_top_n(n):
    def top_n(iterable, n, *, key=None, compare=None):
        def shift(arr, index):
            cout = arr[-1]
            for i in reversed(range(index, len(arr) - 1)):
                arr[i + 1] = arr[i]
            return cout

        if key is None:
            key = lambda x: x
        if compare is None:
            compare = lambda x, y: x > y
        top_values = [None for _ in range(n)]
        for value in iterable:
            for i, top_value in enumerate(top_values):
                if top_value is None or compare(key(value), key(top_value)):
                    shift(top_values, i)
                    top_values[i] = value
                    break
        return top_values

    results = dumpers.json_load("resources/analyser/results.json")
    top = top_n(results, n, key=lambda x: Score.value_of(x["scores"]["codes"]).F_score(1))
    for result in top:
        options = result["options"]
        losses = result["losses"]
        scores = result["scores"]
        dumpers.json_print(options)
        dumpers.json_print(losses)
        dumpers.json_print(scores)
        names = ("labels", "tokens", "strings", "templates", "codes")
        print_scores(Score.value_of(scores[name]) for name in names)


def hist(x, y, n_bins):
    x = np.asarray(x)
    y = np.asarray(y)
    size = (np.max(x) - np.min(x)) / n_bins
    start = np.min(x)
    bins = [start + size * i for i in range(n_bins)]
    height = lambda a, b: np.mean(y[np.logical_and(x >= a, x <= b)])
    heights = [height(start + size * i, start + size * (i + 1)) for i in range(n_bins)]
    plt.bar(bins, heights, size, facecolor='g', alpha=0.2, edgecolor='black', align='edge')
    plt.xticks(bins + [np.max(x)])
    plt.grid(True)


def net_params():
    names = (
        "inputs_state_size",
        "labels_state_size",
        "tokens_state_size",
        "strings_state_size",
        "inputs_hidden_size",
        "labels_hidden_size",
        "tokens_hidden_size",
        "strings_hidden_size"
    )
    results = dumpers.json_load("resources/analyser/results.json")
    params = tuple(zip(*(sub(res["options"]) for res in results)))
    y = [Score.value_of(res["scores"]["codes"]).F_score(1) for res in results]
    return params, y, names


def random_samples(num_samples, ideal_params, rng):
    low, high = rng
    ideal_params = np.asarray(ideal_params)
    samples = [np.random.uniform(low, high, num_samples) for _ in ideal_params]
    evaluate = lambda x: 1 / np.linalg.norm(x - ideal_params)
    y = [evaluate(sample) for sample in np.asarray(samples).T]
    names = tuple(chr(ord('A') + i) for i in range(len(ideal_params)))
    return samples, y, names


def best_params(params, y, n_bins):
    best = []
    for x in params:
        assert len(x) == len(y)
        x = np.asarray(x)
        y = np.asarray(y)
        start = np.min(x)
        size = (np.max(x) - np.min(x)) / n_bins
        height = lambda a, b: np.mean(y[np.logical_and(x >= a, x <= b)])
        heights = [height(start + size * i, start + size * (i + 1)) for i in range(n_bins)]
        # heights = np.asarray(heights)
        # heights = heights / np.sum(heights)
        bins = [start + size * i for i in range(n_bins)]
        bins = np.asarray(bins)
        best_bin, _ = max(zip(bins, heights), key=lambda x: x[1])
        best.append(best_bin + size / 2)
    return best


def print_average_params(n_bins):
    params, y, names = net_params()
    # params, y, names = random_samples(int(1e6), [10, 20, 100, 150], [0, 200])
    estimated_params = best_params(params, y, n_bins)
    dumpers.json_print(dict(zip(names, estimated_params)))
    for x, name in zip(params, names):
        plt.figure(name)
        hist(x, y, n_bins)
    plt.show()


if __name__ == '__main__':
    print_top_n(3)
    # print_average_params(10)
