import random
from multiprocessing import Pool

from seq2seq import contracts
from seq2seq.Net import *
from seq2seq.seq2seq import *
from utils import batcher, dumper
from utils.Figure import Figure
from utils.wrapper import *
from variables.embeddings import *
from variables.path import *
from variables.sintax import *
from variables.tags import *
from variables.train import *


class MünchhausenNet(Net):
    def __init__(self):
        super().__init__("munchhausen")
        self.indexes = None
        self.inputs = None
        self.inputs_sizes = None
        self.output = None
        self.evaluation = None
        self.q = None
        self.diff = None
        self.losses = None
        self.prev_diff = None
        self.prev_q = None
        self._analyser_variables = None
        self.analyser_scope = None
        self._q_function_variables = None
        self.q_function_scope = None

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
        for label in PARTS:
            with vs.variable_scope(label):
                self.indexes[label] = []
                for i in range(INPUT_SIZE):
                    placeholder = tf.placeholder(tf.int32, [BATCH_SIZE], "indexes")
                    self.indexes[label].append(placeholder)
                self.inputs_sizes[label] = tf.placeholder(tf.int32, [BATCH_SIZE], "input_sizes")
        self.evaluation = tf.placeholder(tf.float32, [BATCH_SIZE], "evaluation")
        embeddings = tf.constant(np.asarray(Embeddings.embeddings()))
        self.inputs = {label: [tf.gather(embeddings, inp) for inp in self.indexes[label]] for label in PARTS}
        self.prev_diff = tf.placeholder(tf.float32, [BATCH_SIZE], "prev_diff")
        self.prev_q = tf.placeholder(tf.float32, [BATCH_SIZE], "prev_q")

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
                    # emb_size = EMBEDDING_SIZE
                    emb_size = NUM_TOKENS
                    goes = tf.zeros([BATCH_SIZE, emb_size])
                    decoder_inputs = [goes] * OUTPUT_SIZE
                    state_size = INPUT_STATE_SIZE
                    initial = INITIAL_STATE
                    logits, _ = build_decoder(decoder_inputs, inputs_states, state_size, initial, emb_size, loop=True)
                # with vs.variable_scope("linear"):
                #     W_shape = [EMBEDDING_SIZE, NUM_TOKENS]
                #     B_shape = [NUM_TOKENS]
                #     W = tf.Variable(initial_value=tf.truncated_normal(W_shape, dtype=tf.float32), name="weights")
                #     B = tf.Variable(initial_value=tf.truncated_normal(B_shape, dtype=tf.float32), name="biases")
                #     logits = tf.reshape(tf.stack(logits), [BATCH_SIZE * OUTPUT_SIZE, EMBEDDING_SIZE])
                #     logits = tf.reshape(tf.matmul(logits, W) + B, [OUTPUT_SIZE, BATCH_SIZE, NUM_TOKENS])
                with vs.variable_scope("softmax"):
                    output = tf.unstack(tf.nn.softmax(logits))
                self.analyser_scope = vs.get_variable_scope().name
            with vs.variable_scope("q-function"):
                with vs.variable_scope("encoder"):
                    fw, bw = build_encoder(tf.unstack(logits), OUTPUT_STATE_SIZE)
                    outp_states = [tf.concat(axis=1, values=[fw[i], bw[-(i + 1)]]) for i in range(OUTPUT_SIZE)]
                with vs.variable_scope("decoder"):
                    Q, _ = build_decoder(outp_states, inputs_states, INPUT_STATE_SIZE, INITIAL_STATE, 1)
                    Q = tf.transpose(tf.reshape(tf.nn.relu(tf.stack(Q)), [OUTPUT_SIZE, BATCH_SIZE]), [1, 0])
                with vs.variable_scope("sigmoid"):
                    W_shape = [OUTPUT_STATE_SIZE * 2, 1]
                    B_shape = [1]
                    W = tf.Variable(initial_value=tf.truncated_normal(W_shape, dtype=tf.float32), name="weights")
                    B = tf.Variable(initial_value=tf.truncated_normal(B_shape, dtype=tf.float32), name="biases")
                    outp_states = tf.reshape(tf.stack(outp_states), [BATCH_SIZE * OUTPUT_SIZE, OUTPUT_STATE_SIZE * 2])
                    I = tf.sigmoid(tf.reshape(tf.matmul(outp_states, W) + B, [BATCH_SIZE, OUTPUT_SIZE]))
                    q = tf.reduce_sum(Q * I, axis=1)
                self.q_function_scope = vs.get_variable_scope().name
            scope = vs.get_variable_scope().name
        self.output = output
        self.q = q
        self.scope = scope

    @trace
    def build_fetches(self):
        with vs.variable_scope("losses"):
            regularization_variables = [self.scope + "/" + variable for variable in REGULARIZATION_VARIABLES]
            l2_loss = build_l2_loss(self.get_variables(), regularization_variables)
            self.diff = self.q - self.evaluation

            diff_loss = tf.sqrt(
                DIFF_WEIGHT * tf.square(self.diff)
                # + Q_WEIGHT * tf.square(self.q)
                # + L2_WEIGHT * tf.square(l2_loss)
                # + Q_INERTNESS * tf.square(self.q - self.prev_q)
            )
            q_loss = tf.sqrt(
                Q_WEIGHT * tf.square(self.q)
                # + DIFF_WEIGHT * tf.square(self.diff)
                # + L2_WEIGHT * tf.square(l2_loss)
                # + DIFF_INERTNESS * tf.square(self.diff - self.prev_diff)
            )
            loss = tf.sqrt(
                Q_WEIGHT * tf.square(self.q)
                + DIFF_WEIGHT * tf.square(self.diff)
                # + L2_WEIGHT * tf.square(l2_loss)
            )
        self.optimisers = {}
        with vs.variable_scope("diff-optimiser"):
            optimisers = {}
            self.optimisers["diff"] = optimisers
            variables = self.get_q_function_variables()
            with vs.variable_scope("adam"):
                optimisers["adam"] = tf.train.AdamOptimizer(1e-3, 0.9).minimize(diff_loss, var_list=variables)
            with vs.variable_scope("adadelta"):
                optimisers["adadelta"] = tf.train.AdadeltaOptimizer(1e-3, 0.95).minimize(diff_loss, var_list=variables)
        with vs.variable_scope("q-optimiser"):
            optimisers = {}
            self.optimisers["q"] = optimisers
            variables = self.get_analyser_variables()
            with vs.variable_scope("adam"):
                optimisers["adam"] = tf.train.AdamOptimizer(1e-3, 0.9).minimize(q_loss, var_list=variables)
            with vs.variable_scope("adadelta"):
                optimisers["adadelta"] = tf.train.AdadeltaOptimizer(1e-3, 0.95).minimize(q_loss, var_list=variables)
        # with vs.variable_scope("mixed-optimiser"):
        #     optimisers = {}
        #     self.optimisers["mixed"] = optimisers
        #     variables = self.get_analyser_variables()
        #     with vs.variable_scope("adam"):
        #         optimisers["adam"] = tf.train.AdamOptimizer(1e-3, 0.9).minimize(loss, var_list=variables)
        #     with vs.variable_scope("adadelta"):
        #         optimisers["adadelta"] = tf.train.AdadeltaOptimizer(1e-3, 0.95).minimize(loss, var_list=variables)
        self.losses = (loss, diff_loss, self.diff, q_loss, self.q)
        self.losses = tuple(tf.reduce_mean(loss) for loss in self.losses)

    @staticmethod
    def build_batch(batch: list):
        indexes = {label: [] for label in PARTS}
        inputs_sizes = {label: [] for label in PARTS}
        for datum in batch:
            for label, _indexes in datum:
                _inputs_sizes = len(_indexes)
                _indexes += [Embeddings.get_index(PAD) for _ in range(INPUT_SIZE - len(_indexes))]
                indexes[label].append(_indexes)
                inputs_sizes[label].append(_inputs_sizes)
        indexes = {label: np.transpose(np.asarray(indexes[label]), axes=(1, 0)) for label in PARTS}
        return indexes, inputs_sizes

    @staticmethod
    def indexes(method):
        doc = []
        for label, (embeddings, text) in method:
            indexes = [Embeddings.get_index(word) for embedding, word in zip(embeddings, text)]
            doc.append((label, indexes))
        return doc

    @trace
    def build_data_set(self):
        methods = dumper.load(VEC_METHODS)
        with Pool() as pool:
            docs = pool.map(MünchhausenNet.indexes, methods)
            docs_baskets = batcher.throwing(docs, [INPUT_SIZE])
            docs = docs_baskets[INPUT_SIZE]
            random.shuffle(docs)
            batches = batcher.chunks(docs, BATCH_SIZE)
            batches = pool.map(MünchhausenNet.build_batch, batches)
        # fixme:
        batches = [batch for batch in batches if len(list(batch[1].values())[0]) == BATCH_SIZE]
        random.shuffle(batches)
        train_set = batches[:int(len(batches) * TRAIN_SET)]
        validation_set = batches[len(train_set):]
        self.train_set = train_set
        self.validation_set = validation_set

    @trace
    def build(self):
        self.build_inputs()
        self.build_outputs()
        self.build_fetches()
        self.build_data_set()

    def epoch(self, session, data_set, optimiser=None):
        losses = []
        values = []

        indexes, inputs_sizes = data_set[0]
        feed_dict = {}
        for label in PARTS:
            feed_dict.update(zip(self.indexes[label], indexes[label]))
            feed_dict[self.inputs_sizes[label]] = inputs_sizes[label]
        fetches = (self.output,)
        output, *_ = session.run(fetches=fetches, feed_dict=feed_dict)
        feed_dict[self.evaluation] = contracts.evaluate(indexes, output)
        fetches = (self.q, self.diff,)
        q, diff, *_ = session.run(fetches=fetches, feed_dict=feed_dict)

        for indexes, inputs_sizes in data_set:
            feed_dict = {}
            for label in PARTS:
                feed_dict.update(zip(self.indexes[label], indexes[label]))
                feed_dict[self.inputs_sizes[label]] = inputs_sizes[label]
            feed_dict[self.evaluation] = contracts.evaluate(indexes, output)
            feed_dict[self.prev_q] = q
            feed_dict[self.prev_diff] = diff
            fetches = (self.output, self.q, self.diff, self.losses,)
            if optimiser is not None:
                fetches += (optimiser,)
            output, q, diff, local_losses, *_ = session.run(fetches=fetches, feed_dict=feed_dict)
            losses.append(local_losses)
            values.append((local_losses, indexes, output))
        return np.mean(losses, axis=(0,)), values

    @trace
    def train(self, restore: bool = False):
        best_train = None
        best_validation = None
        for i in range(MÜNCHHAUSEN_RUNS):
            with tf.Session() as session, \
                    tf.device('/cpu:0'), \
                    Figure(xauto=True, xlabel="epoch", ylabel="mixed-loss") as mixed_loss_figure, \
                    Figure(xauto=True, xlabel="epoch", ylabel="q-loss") as q_loss_figure, \
                    Figure(xauto=True, xlabel="epoch", ylabel="diff-loss") as diff_loss_figure:
                session.run(tf.global_variables_initializer())
                self.reset(session)
                writer = tf.summary.FileWriter(self.path, session.graph)
                writer.close()
                logging.info("run {} with model '{}'".format(i, self.path))
                if restore:
                    self.restore(session)
                size = 13
                f1 = "{{:^{size:d}s}}"
                f2 = "{{:^{size:d}d}}"
                f3 = "{{:^{size:d}.4f}}"
                line1 = ("║" + "│".join((f1,) * 4) + "║" + ("│".join((f1,) * 5) + "║") * 2).format(size=size)
                line2 = ("║" + f2 + "│" + f3 + "│" + f1 + "│" + f1 + "║" + ("│".join((f3,) * 5) + "║") * 2).format(
                    size=size)
                f4 = "─" * size
                line3 = ("╟" + "│".join((f4,) * 4) + "╫" + "│".join((f4,) * 5) + "╫" + "│".join((f4,) * 5) + "╢")
                rows = ("epoch", "time", "loss-name", "optimiser", *(("loss", "diff-loss", "diff", "q-loss", "q") * 2))
                head = line1.format(*rows)
                optimiser_selector = get_optimiser_selector()
                for epoch in range(MÜNCHHAUSEN_EPOCHS):
                    loss_name, optimiser_name = next(optimiser_selector)
                    if epoch % 10 == 0:
                        if epoch > 0:
                            logging.info(line3)
                        logging.info(head)
                        logging.info(line3)
                    optimiser = self.optimisers[loss_name][optimiser_name]
                    clock = time.time()
                    train_loss, *_ = self.epoch(session, self.get_train_set(), optimiser)
                    validation_loss, *_ = self.epoch(session, self.get_validation_set())
                    delay = time.time() - clock
                    logging.info(line2.format(epoch, delay, loss_name, optimiser_name, *train_loss, *validation_loss))
                    mixed_loss_figure.plot(epoch, train_loss[0], ".b")
                    mixed_loss_figure.plot(epoch, validation_loss[0], ".r")
                    q_loss_figure.plot(epoch, train_loss[3], ".b")
                    q_loss_figure.plot(epoch, validation_loss[3], ".r")
                    diff_loss_figure.plot(epoch, train_loss[1], ".b")
                    diff_loss_figure.plot(epoch, validation_loss[1], ".r")
                    self.save(session)
                if best_train is None or best_train[0] > train_loss[0]:
                    best_train = (train_loss[0], self.path)
                if best_validation is None or best_validation[0] > validation_loss[0]:
                    best_validation = (validation_loss[0], self.path)
                # noinspection PyArgumentList
                logging.info("BEST TRAIN: {} {}".format(*best_train))
                # noinspection PyArgumentList
                logging.info("BEST VALIDATION: {} {}".format(*best_validation))

    @trace
    def test(self):
        with tf.Session() as session, tf.device('/cpu:0'):
            session.run(tf.global_variables_initializer())
            self.restore(session)
            train_loss, values = self.epoch(session, self.get_train_set())
            logging.info(train_loss)
            maximum = [float("-inf"), ""]
            for losses, indexes, output in values:
                evaluation = contracts.evaluate(indexes, output)
                output = np.transpose(output, axes=(1, 0, 2))
                for outp, evltn in zip(output, evaluation):
                    contract = " ".join((Tokens.get(np.argmax(soft)).name for soft in outp))
                    logging.info("{:10.4f} {}".format(evltn, contract))
                    if maximum[0] < evltn:
                        maximum[0] = evltn
                        maximum[1] = contract
            logging.info("\n{:10.4f} {}".format(*maximum))

    @staticmethod
    @trace
    def run(foo: str):
        münchhausen_net = MünchhausenNet()
        münchhausen_net.build()
        if foo == "train":
            münchhausen_net.train()
        elif foo == "restore":
            münchhausen_net.train(True)
        elif foo == "test":
            münchhausen_net.test()
