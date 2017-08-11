from multiprocessing import Value
from typing import Iterable, Tuple

from contracts.Compiler import DfsCompiler

from boot import prepares
from configurations.fields import CONTRACT
from configurations.logger import info_logger
from configurations.paths import ANALYSER_DATA_SET, ANALYSER_RAW_DATA_SET
from seq2seq import Embeddings
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
        names = Embeddings.tokens().idx2name
        self._token_counters = {name: Counter(0) for name in names}

    def considers(self, methods: Iterable[dict]) -> Iterable[dict]:
        return (self.consider(method) for method in methods)

    def consider(self, method):
        self._methods_counter.increment()
        tree = method[CONTRACT]
        tokens = DfsCompiler().accept(tree)
        for token in tokens:
            name = prepares.convert(token)
            if name is not None:
                self._token_counters[name].increment()
        return method

    def get_tokens(self) -> Iterable[Tuple[str, int]]:
        return ((name, counter.get_value()) for name, counter in self._token_counters.items())

    def get_num_tokens(self) -> int:
        return sum(counter.get_value() for counter in self._token_counters.values())

    def get_num_methods(self) -> int:
        return self._methods_counter.get_value()


class Statistic:
    def __init__(self):
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
        formatter.print_head()
        formatter.print("methods", *(step.get_num_methods() for step in self.get_accountants()))
        formatter.print("tokens", *(step.get_num_tokens() for step in self.get_accountants()))
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
    with Timer("LOADING"):
        methods = dumpers.json_load(ANALYSER_RAW_DATA_SET)
    methods = prepares.java_doc(methods)
    methods = prepares.contract(methods)
    methods = statistic.accountant("number").considers(methods)
    batches = prepares.batches(methods)
    with Timer("DUMPING"):
        dumpers.pkl_dump(batches, ANALYSER_DATA_SET)
    info_logger.info("Number of batches: %d" % len(batches))
    statistic.show()
    return batches


if __name__ == '__main__':
    prepare_data_set()
