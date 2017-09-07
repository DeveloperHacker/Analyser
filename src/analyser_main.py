from random import Random, randint
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
    methods = dumpers.json_load(RAW_DATA_SET_PATH)
    methods = prepares.java_doc(methods)
    methods = prepares.contract(methods)
    methods = statistic.accountant("number").considers(methods)
    batches = prepares.batches(methods, options.batch_size, FILLING, options.flatten_type)
    data_set = prepares.part(batches, TRAIN_PART, VALIDATION_PART, TEST_PART, Random(SEED))
    dumpers.pkl_dump(data_set, DATA_SET_PATH)
    logger.info("Test set length: %d" % len(data_set.test))
    logger.info("Train set length: %d" % len(data_set.train))
    logger.info("Validation set length: %d" % len(data_set.validation))
    statistic.show()
    return batches


@trace("TRAIN")
def train(options: Options):
    tf.reset_default_graph()
    data_set = dumpers.pkl_load(DATA_SET_PATH)
    net = AnalyserNet(options, data_set)
    if FLAGS.cross:
        net.cross()
    else:
        net.train()


@trace("TEST")
def test(options: Options):
    tf.reset_default_graph()
    data_set = dumpers.pkl_load(DATA_SET_PATH)
    net = AnalyserNet(options, data_set)
    return net.test()


def sub(x):
    defaults = {
        "inputs_hidden_size": "inputs_state_size",
        "labels_hidden_size": "labels_state_size",
        "tokens_hidden_size": "tokens_state_size",
        "strings_hidden_size": "strings_state_size"}
    names = (
        "inputs_state_size",
        "labels_state_size",
        "tokens_state_size",
        "strings_state_size",
        "inputs_hidden_size",
        "labels_hidden_size",
        "tokens_hidden_size",
        "strings_hidden_size",
        "l2_weight")
    params = (x[name] for name in names if name in x)
    params = (x[defaults[name]] if param is None else param for name, param in zip(names, params))
    return tuple(params)


def model_dir(options):
    params = "-".join(str(val) for val in sub(options.serialize())[:8])
    return 'resources/analyser/model-%s' % params


@trace("RANDOM")
def random_options(options: Options):
    results = dumpers.json_load(RESULTS_PATH)
    used_options = [sub(result["options"]) for result in results]
    while sub(options.serialize()) in used_options:
        options.inputs_state_size = randint(5, 15) * 10
        options.labels_state_size = randint(5, 15) * 10
        options.tokens_state_size = randint(5, 15) * 10
        options.strings_state_size = randint(5, 15) * 10
        # options.inputs_hidden_size = randint(5, 15) * 10
        # options.labels_hidden_size = randint(5, 15) * 10
        # options.tokens_hidden_size = randint(5, 15) * 10
        # options.strings_hidden_size = randint(5, 15) * 10
    options.model_dir = model_dir(options)


@trace("STORE")
def store(losses, scores, options: Options):
    labels_loss, tokens_loss, strings_loss, loss = losses
    labels_score, tokens_score, strings_score, templates_score, codes_score = scores
    result = {
        "options": options.serialize(),
        "losses": {
            "labels": float(labels_loss),
            "tokens": float(tokens_loss),
            "strings": float(strings_loss),
            "complex": float(loss)
        },
        "scores": {
            "labels": labels_score.serialize(),
            "tokens": tokens_score.serialize(),
            "strings": strings_score.serialize(),
            "templates": templates_score.serialize(),
            "codes": codes_score.serialize()
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
flags.DEFINE_bool('cross', False, '')
flags.DEFINE_bool('test', False, '')
FLAGS = flags.FLAGS

RESULTS_PATH = "resources/analyser/results.json"
RAW_DATA_SET_PATH = 'resources/data-sets/joda-time.json'
DATA_SET_PATH = 'resources/analyser/data-set.pickle'
FILLING = True
SEED = 58645646
TRAIN_PART = 0.8
VALIDATION_PART = 0.1
TEST_PART = 0.1


def main():
    try:
        options = Options()
        options.l2_weight = 0.001
        options.epochs = 100
        options.batch_size = 4
        options.summaries_dir = 'resources/analyser/summaries'
        options.tokens_output_type = "tree"
        options.flatten_type = "bfs"
        options.label_confidence = 0
        options.token_confidence = 0
        options.string_confidence = 0
        options.inputs_state_size = 50
        options.labels_state_size = 50
        options.tokens_state_size = 50
        options.strings_state_size = 50
        options.model_dir = model_dir(options)
        if FLAGS.random: random_options(options)
        options.validate()
        dumpers.json_print(options.serialize(), logger.error)
        if FLAGS.prepare: prepare(options)
        if FLAGS.train: train(options)
        if FLAGS.test: store(*test(options), options)
    except Exception as ex:
        logger.exception(ex)


if __name__ == '__main__':
    main()
