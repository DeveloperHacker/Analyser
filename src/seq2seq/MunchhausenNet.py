from collections import namedtuple

from seq2seq import contracts
from seq2seq.DataSetBuilder import DataSetBuilder
from seq2seq.MunchhausenFormater import MunchhausenFormatter
from seq2seq.MunchhausenOptimiser import MunchhausenTrainOptimiser, MunchhausenPreTrainOptimiser
from seq2seq.Net import *
from seq2seq.seq2seq import *
from utils import dumper
from live_plotter.Figure import Figure
from utils.wrapper import *
from variables.embeddings import *
from variables.path import *
from variables.syntax import *
from variables.tags import *
from variables.train import *

PreTrainLosses = namedtuple("PreTrainLosses", ["loss", "q_diff", "sample_diff"])
TrainLosses = namedtuple("TrainLosses", ["loss", "q", "q_diff", "evaluation"])


class MunchhausenNet(Net):
    def __init__(self):
        super().__init__("munchhausen")
        self.sample = None  # type: list
        self.indexes = None  # type: list
        self.inputs = None  # type: list
        self.inputs_sizes = None  # type: list
        self.output_logits = None  # type: list
        self.output = None  # type: list
        self.evaluation = None  # type: tf.Tensor
        self.q = None  # type: tf.Tensor
        self.diff = None  # type: tf.Tensor

        self.analyser_scope = None  # type: str
        self.q_function_scope = None  # type: str
        self._analyser_variables = None  # type: list
        self._q_function_variables = None  # type: list

        self.train_losses = None  # type: TrainLosses
        self.pretrain_losses = None  # type: TrainLosses
        self.train_optimizer = None  # type: MunchhausenTrainOptimiser
        self.pretrain_optimizer = None  # type: MunchhausenPreTrainOptimiser

    def get_analyser_variables(self) -> list:
        if self._analyser_variables is None:
            self._analyser_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.analyser_scope)
        return self._analyser_variables

    def get_q_function_variables(self) -> list:
        if self._q_function_variables is None:
            self._q_function_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.q_function_scope)
        return self._q_function_variables

    @trace
    def build_inputs(self):
        self.indexes = {}
        self.inputs_sizes = {}
        self.sample = []
        for label in PARTS:
            with vs.variable_scope(label):
                self.indexes[label] = []
                for i in range(INPUT_SIZE):
                    placeholder = tf.placeholder(tf.int32, [BATCH_SIZE], "indexes")
                    self.indexes[label].append(placeholder)
                self.inputs_sizes[label] = tf.placeholder(tf.int32, [BATCH_SIZE], "input_sizes")
        self.evaluation = tf.placeholder(tf.float32, [BATCH_SIZE], "evaluation")
        for i in range(OUTPUT_SIZE):
            placeholder = tf.placeholder(tf.int32, [BATCH_SIZE, NUM_TOKENS], "sample")
            self.sample.append(placeholder)
        embeddings = tf.constant(np.asarray(Embeddings.embeddings()))
        self.inputs = {label: [tf.gather(embeddings, inp) for inp in self.indexes[label]] for label in PARTS}

    @trace
    def build_outputs(self):
        inputs = self.inputs
        inputs_sizes = self.inputs_sizes
        with vs.variable_scope("munchhausen"):
            with vs.variable_scope("analyser"):
                with vs.variable_scope("encoder"):
                    inputs_states = []
                    for label in PARTS:
                        with vs.variable_scope(label):
                            fw, bw = build_encoder(inputs[label], INPUT_STATE_SIZE, inputs_sizes[label])
                            inputs_states.append(tf.concat(axis=1, values=[fw[0], bw[-1]]))
                            inputs_states.append(tf.concat(axis=1, values=[fw[-1], bw[0]]))
                    inputs_states = tf.transpose(tf.stack(inputs_states), [1, 0, 2])
                with vs.variable_scope("decoder"):
                    emb_size = NUM_TOKENS
                    goes = tf.zeros([BATCH_SIZE, emb_size])
                    decoder_inputs = [goes] * OUTPUT_SIZE
                    state_size = INPUT_STATE_SIZE
                    initial = INITIAL_STATE
                    logits, _ = build_decoder(decoder_inputs, inputs_states, state_size, initial, emb_size, loop=True)
                with vs.variable_scope("softmax"):
                    output = tf.unstack(tf.nn.softmax(logits))
                self.analyser_scope = vs.get_variable_scope().name
            with vs.variable_scope("q-function"):
                with vs.variable_scope("encoder"):
                    fw, bw = build_encoder(tf.unstack(logits), OUTPUT_STATE_SIZE)
                    states = [tf.concat(axis=1, values=[fw[i], bw[-(i + 1)]]) for i in range(OUTPUT_SIZE)]
                with vs.variable_scope("decoder"):
                    Q, _ = build_decoder(states, inputs_states, INPUT_STATE_SIZE, INITIAL_STATE, 1)
                    Q = tf.transpose(tf.reshape(tf.nn.relu(tf.stack(Q)), [OUTPUT_SIZE, BATCH_SIZE]), [1, 0])
                with vs.variable_scope("sigmoid"):
                    W_shape = [OUTPUT_STATE_SIZE * 2, 1]
                    B_shape = [1]
                    W = tf.Variable(initial_value=tf.truncated_normal(W_shape, dtype=tf.float32), name="weights")
                    B = tf.Variable(initial_value=tf.truncated_normal(B_shape, dtype=tf.float32), name="biases")
                    states = tf.reshape(tf.stack(states), [BATCH_SIZE * OUTPUT_SIZE, OUTPUT_STATE_SIZE * 2])
                    I = tf.sigmoid(tf.reshape(tf.matmul(states, W) + B, [BATCH_SIZE, OUTPUT_SIZE]))
                    q = tf.reduce_sum(Q * I, axis=1)
                self.q_function_scope = vs.get_variable_scope().name
            scope = vs.get_variable_scope().name
        self.output_logits = logits
        self.output = output
        self.q = q
        self.scope = scope

    @trace
    def build_fetches(self):
        with vs.variable_scope("losses"):
            self.diff = self.q - self.evaluation
            q_diff_loss = tf.abs(self.diff)
            q_loss = self.q
            sample_indexes = tf.transpose(tf.argmax(self.sample, 2), (1, 0))
            logits = tf.transpose(self.output_logits, (1, 0, 2))
            sample_diff_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=sample_indexes, logits=logits)
            loss = tf.sqrt(q_diff_loss + q_loss)
        self.pretrain_optimizer = MunchhausenPreTrainOptimiser(self, q_diff_loss, sample_diff_loss)
        self.train_optimizer = MunchhausenTrainOptimiser(self, q_loss, q_diff_loss)
        self.train_losses = (loss, q_loss, q_diff_loss, self.evaluation)
        self.train_losses = TrainLosses(*(tf.reduce_mean(loss) for loss in self.train_losses))
        self.pretrain_losses = (loss, q_diff_loss, sample_diff_loss)
        self.pretrain_losses = PreTrainLosses(*(tf.reduce_mean(loss) for loss in self.pretrain_losses))

    @staticmethod
    @trace
    def build_data_set():
        DataSetBuilder.build_best_data_set(MUNCHHAUSEN_PRETRAIN_SET, 20, 20, 3)
        DataSetBuilder.build_vectorized_methods_data_set(MUNCHHAUSEN_TRAIN_SET)

    @trace
    def load_data_set(self, save_path: str):
        batches = dumper.load(save_path)
        self.train_set = batches[:int(len(batches) * TRAIN_SET)]
        self.validation_set = batches[-int(len(batches) * VALIDATION_SET):]

    @trace
    def build(self):
        self.build_inputs()
        self.build_outputs()
        self.build_fetches()
        self.load_data_set(MUNCHHAUSEN_TRAIN_SET)

    def pretrain_epoch(self, session, data_set, optimizers: list = None):
        losses = []
        values = []
        for i, (indexes, inputs_sizes, sample, evaluation) in enumerate(data_set):
            feed_dict = {}
            for label in PARTS:
                feed_dict.update(zip(self.indexes[label], indexes[label]))
                feed_dict[self.inputs_sizes[label]] = inputs_sizes[label]
            feed_dict[self.evaluation] = evaluation
            feed_dict.update(zip(self.sample, sample))
            fetches = (self.output, self.pretrain_losses)
            if optimizers is not None:
                fetches += optimizers
            output, local_losses, *_ = session.run(fetches=fetches, feed_dict=feed_dict)
            losses.append(local_losses)
            values.append((local_losses, indexes, output))
        return PreTrainLosses(*np.mean(losses, axis=0)), values

    def train_epoch(self, session, data_set, optimizers: list = None):
        losses = []
        values = []
        for i, (indexes, inputs_sizes) in enumerate(data_set):
            feed_dict = {}
            for label in PARTS:
                feed_dict.update(zip(self.indexes[label], indexes[label]))
                feed_dict[self.inputs_sizes[label]] = inputs_sizes[label]
            if i == 0:
                fetches = (self.output,)
                output, *_ = session.run(fetches=fetches, feed_dict=feed_dict)
            else:
                feed_dict[self.evaluation] = contracts.evaluate(indexes, output)
                fetches = (self.output, self.train_losses)
                if optimizers is not None:
                    fetches += optimizers
                output, local_losses, *_ = session.run(fetches=fetches, feed_dict=feed_dict)
                losses.append(local_losses)
                values.append((local_losses, indexes, output))
        return TrainLosses(*np.mean(losses, axis=0)), values

    @staticmethod
    def variable_summaries(variable: tf.Variable):
        with vs.variable_scope(variable.name[:-2].replace("/", "--")):
            mean, var = tf.nn.moments(variable, tuple(i for i in range(len(variable.shape))))
            tf.summary.scalar('mean', mean)
            tf.summary.scalar('stddev', tf.sqrt(var))
            tf.summary.scalar('max', tf.reduce_max(variable))
            tf.summary.scalar('is_nan', tf.cast(tf.is_nan(mean), tf.float32))

    @trace
    def run(self):
        best_train = (float("inf"), "")
        best_validation = (float("inf"), "")
        best_pretrain = (float("inf"), "")
        best_prevalidation = (float("inf"), "")
        worst_train = (float("-inf"), "")
        worst_validation = (float("-inf"), "")
        worst_pretrain = (float("-inf"), "")
        worst_prevalidation = (float("-inf"), "")
        for i in range(MÜNCHHAUSEN_RUNS):
            self.load_data_set(MUNCHHAUSEN_PRETRAIN_SET)
            pretrain_loss, prevalidation_loss = self.pretrain(MÜNCHHAUSEN_PRETRAIN_EPOCHS)
            self.load_data_set(MUNCHHAUSEN_TRAIN_SET)
            train_loss, validation_loss = self.train(MÜNCHHAUSEN_TRAIN_EPOCHS, restore=True)
            if best_train[0] > train_loss: best_train = (train_loss, self.path)
            if best_validation[0] > validation_loss: best_validation = (validation_loss, self.path)
            if best_pretrain[0] > train_loss: best_pretrain = (pretrain_loss, self.path)
            if best_prevalidation[0] > validation_loss: best_prevalidation = (prevalidation_loss, self.path)
            if worst_train[0] < train_loss: worst_train = (train_loss, self.path)
            if worst_validation[0] < validation_loss: worst_validation = (validation_loss, self.path)
            if worst_pretrain[0] < train_loss: worst_pretrain = (pretrain_loss, self.path)
            if worst_prevalidation[0] < validation_loss: worst_prevalidation = (prevalidation_loss, self.path)
            logging.info("BEST train loss = {}: {}".format(*best_train))
            logging.info("BEST validation loss = {}: {}".format(*best_validation))
            logging.info("BEST pretrain loss = {}: {}".format(*best_pretrain))
            logging.info("BEST prevalidation loss = {}: {}".format(*best_prevalidation))
            logging.info("WORST train loss = {}: {}".format(*worst_train))
            logging.info("WORST validation loss = {}: {}".format(*worst_validation))
            logging.info("WORST pretrain loss = {}: {}".format(*worst_pretrain))
            logging.info("WORST prevalidation loss = {}: {}".format(*worst_prevalidation))

    @trace
    def pretrain(self, epochs: int):
        formatter = MunchhausenFormatter(("epoch", "time"), ("loss", "q-diff", "sample-diff"))
        Figure.ion()
        figure = Figure("pretrain")
        train_loss_graph = figure.curve(3, 1, 1, mode="-ob")
        validation_loss_graph = figure.curve(3, 1, 1, mode="-or")
        train_q_diff_graph = figure.curve(3, 1, 2, mode="-ob")
        validation_q_diff_graph = figure.curve(3, 1, 2, mode="-or")
        train_sample_diff_graph = figure.curve(3, 1, 3, mode="-ob")
        validation_sample_diff_graph = figure.curve(3, 1, 3, mode="-or")
        figure.set_label(3, 1, 1, "loss")
        figure.set_label(3, 1, 2, "q-diff")
        figure.set_label(3, 1, 3, "sample-diff")
        figure.set_x_label(3, 1, 1, "epoch")
        figure.set_x_label(3, 1, 2, "epoch")
        figure.set_x_label(3, 1, 3, "epoch")
        figure.set_y_label(3, 1, 1, "loss")
        figure.set_y_label(3, 1, 2, "loss")
        figure.set_y_label(3, 1, 3, "loss")
        with tf.Session() as session, tf.device('/cpu:0'):
            self.reset(session)
            self.mkdir()
            formatter.run(self.path)
            for epoch in range(epochs):
                optimizers = self.pretrain_optimizer.get_tf_optimizers()
                clock = time.time()
                train_loss, *_ = self.pretrain_epoch(session, self.get_train_set(), optimizers)
                validation_loss, *_ = self.pretrain_epoch(session, self.get_validation_set())
                delay = time.time() - clock
                self.save(session)
                formatter.print(epoch, delay, *train_loss, *validation_loss)
                train_loss_graph.append(epoch, train_loss.loss)
                validation_loss_graph.append(epoch, validation_loss.loss)
                train_q_diff_graph.append(epoch, train_loss.q_diff)
                validation_q_diff_graph.append(epoch, validation_loss.q_diff)
                train_sample_diff_graph.append(epoch, train_loss.sample_diff)
                validation_sample_diff_graph.append(epoch, validation_loss.sample_diff)
                figure.draw()
                figure.save(self.path + "/pretrain.png")
        return train_loss.loss, validation_loss.loss

    @trace
    def train(self, epochs: int, restore: bool = False) -> (float, float):
        formatter = MunchhausenFormatter(("epoch", "time"), ("loss", "q", "q-diff", "eval"))
        Figure.ion()
        figure = Figure("train")
        train_loss_graph = figure.curve(3, 1, 1, mode="-ob")
        validation_loss_graph = figure.curve(3, 1, 1, mode="-or")
        train_q_diff_graph = figure.curve(3, 1, 2, mode="-ob")
        validation_q_diff_graph = figure.curve(3, 1, 2, mode="-or")
        train_q_loss_graph = figure.curve(3, 1, 3, mode="-ob")
        validation_q_loss_graph = figure.curve(3, 1, 3, mode="-or")
        figure.set_label(3, 1, 1, "loss")
        figure.set_label(3, 1, 2, "q-diff")
        figure.set_label(3, 1, 3, "q-loss")
        figure.set_x_label(3, 1, 1, "epoch")
        figure.set_x_label(3, 1, 2, "epoch")
        figure.set_x_label(3, 1, 3, "epoch")
        figure.set_y_label(3, 1, 1, "loss")
        figure.set_y_label(3, 1, 2, "loss")
        figure.set_y_label(3, 1, 3, "loss")
        with tf.Session() as session, tf.device('/cpu:0'):
            self.reset(session)
            if restore:
                self.restore(session)
            else:
                self.mkdir()
            formatter.run(self.path)
            for epoch in range(epochs):
                optimizers = self.train_optimizer.get_tf_optimizers()
                clock = time.time()
                train_loss, *_ = self.train_epoch(session, self.get_train_set(), optimizers)
                validation_loss, *_ = self.train_epoch(session, self.get_validation_set())
                delay = time.time() - clock
                self.train_optimizer.update(train_loss.q_diff)
                self.save(session)
                formatter.print(epoch, delay, *train_loss, *validation_loss)
                train_loss_graph.append(epoch, train_loss.loss)
                validation_loss_graph.append(epoch, validation_loss.loss)
                train_q_diff_graph.append(epoch, train_loss.q_diff)
                validation_q_diff_graph.append(epoch, validation_loss.q_diff)
                train_q_loss_graph.append(epoch, train_loss.q)
                validation_q_loss_graph.append(epoch, validation_loss.q)
                figure.draw()
                figure.save(self.path + "/train.png")
        return train_loss.loss, validation_loss.loss

    @trace
    def test(self):
        with tf.Session() as session, tf.device('/cpu:0'):
            self.reset(session)
            self.restore(session)
            train_loss, values = self.train_epoch(session, self.get_train_set())
            logging.info(train_loss)
            maximum = [float("-inf"), ""]
            minimum = [float("inf"), ""]
            for losses, indexes, output in values:
                evaluation = contracts.evaluate(indexes, output)
                output = np.transpose(output, axes=(1, 0, 2))
                for _output, _evaluation in zip(output, evaluation):
                    contract = " ".join((Tokens.get(np.argmax(soft)).name for soft in _output))
                    logging.info("{:10.4f} {:10.4f} {}".format(_evaluation, losses.q, contract))
                    if maximum[0] < _evaluation:
                        maximum[0] = _evaluation
                        maximum[1] = contract
                    if minimum[0] > _evaluation:
                        minimum[0] = _evaluation
                        minimum[1] = contract
            logging.info("\nBEST  {:10.4f} {}".format(*minimum))
            logging.info("WORST {:10.4f} {}\n".format(*maximum))

    @staticmethod
    @trace
    def start(foo: str):
        münchhausen_net = MunchhausenNet()
        münchhausen_net.build()
        if foo == "run":
            münchhausen_net.run()
        elif foo == "train":
            münchhausen_net.train(MÜNCHHAUSEN_TRAIN_EPOCHS)
        elif foo == "restore":
            münchhausen_net.train(MÜNCHHAUSEN_TRAIN_EPOCHS, True)
        elif foo == "test":
            münchhausen_net.test()
