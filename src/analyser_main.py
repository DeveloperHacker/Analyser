import random
from typing import Iterable, Tuple

import tensorflow as tf
from contracts.Compiler import DfsCompiler

import prepares
from analyser import Embeddings
from analyser.AnalyserNet import AnalyserNet
from analyser.Options import Options
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
        names = Embeddings.tokens().idx2name + Embeddings.labels().idx2name
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
def prepare(options: Options):
    statistic = Statistic()
    methods = prepares.load(RAW_DATA_SET_PATH)
    methods = prepares.java_doc(methods)
    methods = prepares.contract(methods)
    methods = statistic.accountant("number").considers(methods)
    batches = prepares.batches(methods, options.batch_size)
    batches = list(batches)
    dumpers.pkl_dump(batches, options.data_set_path)
    logger.info("Number of batches: %d" % len(batches))
    statistic.show()
    return batches


@trace("TRAIN")
def train(options: Options):
    tf.reset_default_graph()
    net = AnalyserNet(options)
    net.train()


@trace("TEST")
def test(options: Options):
    tf.reset_default_graph()
    net = AnalyserNet(options)
    return net.test()


@trace("RANDOM")
def random_options(options: Options):
    results = dumpers.json_load(RESULTS_PATH)
    used_options = [result["options"] for result in results]
    while options.serialize() in used_options:
        options.inputs_state_size = random.randint(1, 20) * 10
        options.inputs_hidden_size = random.randint(1, 20) * 10
        options.tokens_state_size = random.randint(1, 20) * 10
        options.strings_state_size = random.randint(1, 20) * 10
    names = ("inputs_state_size", "labels_state_size", "tokens_state_size", "strings_state_size")
    sub = lambda x: tuple((name, x[name]) for name in names if name in x)
    options.model_dir = 'resources/analyser/model-%d-%d-%d-%d' % tuple(x[1] for x in sub(options.serialize()))


@trace("STORE")
def store(losses, scores, options: Options):
    labels_loss, tokens_loss, strings_loss, loss = losses
    labels_score, tokens_score, strings_score, templates_score, codes_score = scores
    result = {
        "options": options.serialize(),
        "scores": {
            "labels_loss": labels_loss,
            "tokens_loss": tokens_loss,
            "strings_loss": strings_loss,
            "loss": loss,
            "labels_score": tokens_score.serialize(),
            "tokens_score": tokens_score.serialize(),
            "strings_score": strings_score.serialize(),
            "templates_score": templates_score.serialize(),
            "codes_score": codes_score.serialize()
        }
    }
    dumpers.json_print(result, logger.error)
    results = dumpers.json_load(RESULTS_PATH)
    results.append(result)
    dumpers.json_dump(results, RESULTS_PATH)


flags = tf.app.flags

flags.DEFINE_bool('prepare', False, '')
flags.DEFINE_bool('random', False, '')
flags.DEFINE_bool('train', False, '')

FLAGS = flags.FLAGS

RESULTS_PATH = "resources/analyser/results.json"
RAW_DATA_SET_PATH = 'resources/data-sets/joda-time.json'


def main():
    options = Options()
    options.minimum_length = 2
    options.train_set = 0.8
    options.validation_set = 0.1
    options.test_set = 0.1
    options.l2_weight = 0.001
    options.epochs = 100
    options.batch_size = 4
    options.summaries_dir = 'resources/analyser/summaries'
    options.data_set_path = 'resources/analyser/data-set.pickle'
    options.inputs_state_size = 40
    options.labels_state_size = 40
    options.tokens_state_size = 40
    options.strings_state_size = 40
    options.model_dir = 'resources/analyser/model-40-40-40-40'
    if FLAGS.random: random_options(options)
    dumpers.json_print(options.serialize(), logger.error)
    if FLAGS.prepare: prepare(options)
    if FLAGS.train: train(options)
    losses, scores = test(options)
    store(losses, scores, options)


if __name__ == '__main__':
    main()
