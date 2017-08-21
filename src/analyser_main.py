import random
from typing import Iterable, Tuple

import tensorflow as tf
from contracts.Compiler import DfsCompiler

import prepares
from analyser import Embeddings
from analyser.AnalyserNet import AnalyserNet
from contants import CONTRACT
from logger import logger
from utils import dumpers
from utils.Formatter import Formatter
from utils.wrappers import trace


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
            token = prepares.convert(token)
            if token is not None:
                self._token_counters[token.name].increment()
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
def prepare():
    statistic = Statistic()
    methods = prepares.load(FLAGS.raw_data_set_path)
    methods = prepares.java_doc(methods)
    methods = prepares.contract(methods)
    methods = statistic.accountant("number").considers(methods)
    batches = prepares.batches(methods, FLAGS.batch_size)
    batches = list(batches)
    dumpers.pkl_dump(batches, FLAGS.data_set_path)
    logger.info("Number of batches: %d" % len(batches))
    statistic.show()
    return batches


@trace("TRAIN")
def train():
    tf.reset_default_graph()
    net = AnalyserNet()
    return net.train()


@trace("TEST")
def test():
    tf.reset_default_graph()
    net = AnalyserNet()
    return net.test()


flags = tf.app.flags

flags.DEFINE_bool('prepare', False, '')
flags.DEFINE_bool('train', False, '')
flags.DEFINE_bool('test', False, '')
flags.DEFINE_bool('random', False, '')

FLAGS = tf.app.flags.FLAGS

FLAGS.minimum_length = 2
FLAGS.train_set = 0.8
FLAGS.validation_set = 0.1
FLAGS.test_set = 0.1
FLAGS.l2_weight = 0.001
FLAGS.epochs = 100
FLAGS.batch_size = 4
FLAGS.input_state_size = 100
FLAGS.input_hidden_size = 100
FLAGS.token_state_size = 100
FLAGS.string_state_size = 100
params = (FLAGS.input_state_size, FLAGS.input_hidden_size, FLAGS.token_state_size, FLAGS.string_state_size)
FLAGS.model_dir = 'resources/analyser/model-%d-%d-%d-%d' % params
FLAGS.summaries_dir = 'resources/analyser/summaries'
FLAGS.data_set_path = 'resources/analyser/data-set.pickle'
FLAGS.raw_data_set_path = 'resources/data-sets/joda-time.json'
FLAGS.parameters_path = "resources/analyser/parameters.json"


def train_with_random_params():
    results = dumpers.json_load(FLAGS.parameters_path)
    used_parameters = [result["parameters"] for result in results]
    while True:
        parameters = {
            "input_state_size": random.randint(1, 10) * 10,
            "input_hidden_size": random.randint(1, 50) * 10,
            "token_state_size": random.randint(1, 50) * 10,
            "string_state_size": random.randint(1, 50) * 10
        }
        if parameters not in used_parameters:
            break
    FLAGS.input_state_size = parameters["input_state_size"]
    FLAGS.input_hidden_size = parameters["input_hidden_size"]
    FLAGS.token_state_size = parameters["token_state_size"]
    FLAGS.string_state_size = parameters["string_state_size"]
    params = (FLAGS.input_state_size, FLAGS.input_hidden_size, FLAGS.token_state_size, FLAGS.string_state_size)
    FLAGS.model_dir = 'resources/analyser/model-%d-%d-%d-%d' % params
    logger.error(str(parameters))
    loss = train()
    scores = test()
    tokens_score, strings_score, templates_score, codes_score = scores
    result = {
        "parameters": parameters,
        "scores": {
            "loss": int(loss),
            "tokens_score": [int(err) for err in tokens_score.tuple()],
            "strings_score": [int(err) for err in strings_score.tuple()],
            "templates_score": [int(err) for err in templates_score.tuple()],
            "codes_score": [int(err) for err in codes_score.tuple()]
        }
    }
    results.append(result)
    dumpers.json_dump(results, FLAGS.parameters_path)


def main():
    if FLAGS.prepare: prepare()
    if FLAGS.random:
        train_with_random_params()
        return
    if FLAGS.train: train()
    if FLAGS.test: test()


if __name__ == '__main__':
    main()
