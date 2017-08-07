import itertools
import random
from multiprocessing import Value
from multiprocessing.pool import Pool
from typing import Iterable, Tuple

from contracts.tokens import Predicates, Markers, Labels

from boot import prepares
from configurations.paths import ANALYSER_DATA_SET, ANALYSER_RAW_DATA_SET
from utils import dumpers
from utils.Formatter import Formatter
from utils.wrappers import Timer, trace


class Counter:
    def set_value(self, value: int):
        self._value = value

    def get_value(self) -> int:
        return self._value

    def __init__(self, initial: int):
        self._value = initial

    def increment(self) -> int:
        self._value += 1
        return self._value


class SynchronizedCounter:
    def set_value(self, value: int):
        self._value.value = value

    def get_value(self) -> int:
        return self._value.value

    def __init__(self, initial: int):
        self._value = Value("i", initial)

    def increment(self) -> int:
        with self._value.get_lock():
            self._value.value += 1
            result = self._value.value
        return result


class Accountant:
    def __init__(self):
        self._methods_counter = Counter(0)
        self._token_counters = {token: Counter(0) for token in itertools.chain(Predicates.names, Markers.names)}
        self._label_counters = {token: Counter(0) for token in Labels.names}

    def considers(self, methods: Iterable[dict]) -> Iterable[dict]:
        return (self.consider(method) for method in methods)

    def consider(self, method):
        self._methods_counter.increment()
        for label, tokens, strings in method["contract"]:
            for token in tokens:
                self._token_counters[token.name].increment()
            self._label_counters[label.name].increment()
        return method

    def get_tokens(self) -> Iterable[Tuple[str, int]]:
        return ((name, counter.get_value()) for name, counter in self._token_counters.items())

    def get_labels(self) -> Iterable[Tuple[str, int]]:
        return ((name, counter.get_value()) for name, counter in self._label_counters.items())

    def get_num_tokens(self) -> int:
        return sum(counter.get_value() for counter in self._token_counters.values())

    def get_num_labels(self):
        return sum(counter.get_value() for counter in self._label_counters.values())

    def get_num_methods(self) -> int:
        return self._methods_counter.get_value()


class Statistic:
    def __init__(self):
        self.num_methods = 0
        self.num_batches = 0
        self.accountant_ids = []
        self._accountants = {}

    def accountant(self, accountant_id: str):
        accountant = Accountant()
        self.accountant_ids.append(accountant_id)
        self._accountants[accountant_id] = accountant
        return accountant

    def get_accountants(self) -> Iterable[Accountant]:
        for accountant_id in self.accountant_ids:
            yield self._accountants[accountant_id]

    def show(self):
        num_accountants = len(self.accountant_ids)
        assert num_accountants > 0
        heads = ("name", *self.accountant_ids)
        formats = ["s"] + ["d"] * num_accountants
        sizes = [15] * (num_accountants + 1)
        formatter = Formatter(heads, formats, sizes, range(num_accountants + 1))
        formatter.raw_print("Number of methods: %d" % self.num_methods)
        formatter.raw_print("Number of batches: %d" % self.num_batches)
        formatter.print_head()
        formatter.print("methods", *(step.get_num_methods() for step in self.get_accountants()))
        formatter.print("labels", *(step.get_num_labels() for step in self.get_accountants()))
        formatter.print("tokens", *(step.get_num_tokens() for step in self.get_accountants()))
        formatter.print_delimiter()
        values = []
        for labels in zip(*(step.get_labels() for step in self.get_accountants())):
            assert all(labels[0][0] == label[0] for label in labels)
            values.append((labels[0][0], *(label[1] for label in labels)))
        for name, *args in sorted(values, key=lambda x: x[1], reverse=True):
            formatter.print(name, *args)
        formatter.print_delimiter()
        values = []
        for tokens in zip(*(step.get_tokens() for step in self.get_accountants())):
            assert all(tokens[0][0] == token[0] for token in tokens)
            values.append((tokens[0][0], *(token[1] for token in tokens)))
        for name, *args in sorted(values, key=lambda x: x[1], reverse=True):
            formatter.print(name, *args)
        formatter.print_lower_delimiter()


@trace("PREPARE DATA-SET")
def prepare_data_set():
    statistic = Statistic()
    methods = dumpers.json_load(ANALYSER_RAW_DATA_SET)
    statistic.num_methods = len(methods)
    with Pool() as pool:
        with Timer("step 1") as timer:
            methods = (method for method in methods if not prepares.empty(method))
            methods = pool.map(prepares.join_java_doc, methods)
            methods = pool.map(prepares.apply_anonymizers, methods)
            methods = (method for method in methods if not prepares.empty(method))
            methods = pool.map(prepares.one_line_doc, methods)
            methods = pool.map(prepares.index_java_doc, methods)
            methods = pool.map(prepares.parse_contract, methods)
            methods = statistic.accountant(timer.name).considers(methods)
        with Timer("step 2") as timer:
            methods = (method for method in methods if len(method["contract"]) > 0)
            methods = pool.map(prepares.filter_contract, methods)
            methods = pool.map(prepares.standardify_contract, methods)
            methods = (method for method in methods if len(method["contract"]) > 0)
            methods = prepares.align_data_set(methods)
            methods = statistic.accountant(timer.name).considers(methods)
        with Timer("step 3"):
            methods = pool.map(prepares.filter_contract_text, methods)
            methods = pool.map(prepares.index_contract, methods)
            batches = prepares.batching(methods)
            batches = pool.map(prepares.build_batch, batches)
            statistic.num_batches = len(batches)
    with Timer("step 4"):
        random.shuffle(batches)
    with Timer("step 5"):
        dumpers.pkl_dump(batches, ANALYSER_DATA_SET)
    statistic.show()
    return batches


if __name__ == '__main__':
    prepare_data_set()
