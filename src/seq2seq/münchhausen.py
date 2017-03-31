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
            with vs.variable_scope("javadoc-encoder"):
                inputs_states = []
                for label in PARTS:
                    with vs.variable_scope(label):
                        fw, bw = build_encoder(inputs[label], INPUT_STATE_SIZE, inputs_sizes[label])
                        inputs_states.append(tf.concat(axis=1, values=[fw[0], bw[-1]]))
                        inputs_states.append(tf.concat(axis=1, values=[fw[-1], bw[0]]))
            inputs_states = tf.transpose(tf.stack(inputs_states), [1, 0, 2])
            goes = tf.stack([Embeddings.get_embedding(GO)] * BATCH_SIZE)
            with vs.variable_scope("contract-decoder"):
                decoder_inputs = [goes] * OUTPUT_SIZE
                state_size = INPUT_STATE_SIZE
                initial = INITIAL_STATE
                emb_size = EMBEDDING_SIZE
                logits, _ = build_decoder(decoder_inputs, inputs_states, state_size, initial, emb_size, loop=True)
            with vs.variable_scope("contract-linear"):
                W_shape = [EMBEDDING_SIZE, NUM_TOKENS]
                B_shape = [NUM_TOKENS]
                W = tf.Variable(initial_value=tf.truncated_normal(W_shape, dtype=tf.float32), name="weights")
                B = tf.Variable(initial_value=tf.truncated_normal(B_shape, dtype=tf.float32), name="biases")
                logits = tf.reshape(tf.stack(logits), [BATCH_SIZE * OUTPUT_SIZE, EMBEDDING_SIZE])
                logits = tf.reshape(tf.matmul(logits, W) + B, [OUTPUT_SIZE, BATCH_SIZE, NUM_TOKENS])
            with vs.variable_scope("contract-softmax"):
                output = tf.unstack(tf.nn.softmax(logits))
            with vs.variable_scope("evaluation-encoder"):
                fw, bw = build_encoder(tf.unstack(logits), OUTPUT_STATE_SIZE)
                output_states = [tf.concat(axis=1, values=[fw[i], bw[-(i + 1)]]) for i in range(OUTPUT_SIZE)]
            with vs.variable_scope("evaluation-decoder"):
                Q, _ = build_decoder(output_states, inputs_states, INPUT_STATE_SIZE, INITIAL_STATE, 1)
                Q = tf.transpose(tf.reshape(tf.nn.relu(tf.stack(Q)), [OUTPUT_SIZE, BATCH_SIZE]), [1, 0])
            with vs.variable_scope("evaluation-sigmoid"):
                W_shape = [OUTPUT_STATE_SIZE * 2, 1]
                B_shape = [1]
                W = tf.Variable(initial_value=tf.truncated_normal(W_shape, dtype=tf.float32), name="weights")
                B = tf.Variable(initial_value=tf.truncated_normal(B_shape, dtype=tf.float32), name="biases")
                output_states = tf.reshape(tf.stack(output_states), [BATCH_SIZE * OUTPUT_SIZE, OUTPUT_STATE_SIZE * 2])
                I = tf.sigmoid(tf.reshape(tf.matmul(output_states, W) + B, [BATCH_SIZE, OUTPUT_SIZE]))
            q = tf.reduce_sum(Q * I, axis=1)
            scope = vs.get_variable_scope().name
        self.output = output
        self.q = q
        self.scope = scope

    @trace
    def build_fetches(self):
        with vs.variable_scope("losses"):
            regularization_variables = [self.scope + "/" + variable for variable in REGULARIZATION_VARIABLES]
            l2_loss = L2_WEIGHT * build_l2_loss(self.get_variables(), regularization_variables)
            self.diff = self.evaluation - self.q
            diff_inertness = self.prev_diff - self.diff
            q_inertness = self.prev_q - self.q
            diff_loss = tf.sqrt(tf.square(self.diff) + tf.square(q_inertness) + tf.square(l2_loss))
            q_loss = tf.sqrt(tf.square(self.q) + tf.square(diff_inertness) + tf.square(l2_loss))
            loss = self.diff + self.q + l2_loss
        with vs.variable_scope("diff-optimiser"):
            diff_optimiser = tf.train.AdamOptimizer(1e-3, 0.9).minimize(diff_loss, var_list=self.get_variables())
        with vs.variable_scope("q-optimiser"):
            q_optimise = tf.train.AdamOptimizer(1e-3, 0.9).minimize(q_loss, var_list=self.get_variables())
        self.optimisers = (diff_optimiser, q_optimise)
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
        for i in range(MÜNCHHAUSEN_RUNS):
            with tf.Session() as session, tf.device('/cpu:0'), Figure(xauto=True) as figure:
                # writer = tf.summary.FileWriter(SEQ2SEQ, session.graph)
                # writer.close()
                session.run(tf.global_variables_initializer())
                self.reset(session)
                logging.info("run {} with model '{}'".format(i, self.path))
                if restore:
                    self.restore(session)
                size = 13
                head1 = ("|{{:^{size1:d}s}}|" + "{{:^{size2:d}s}}|" * 2).format(size1=size * 2 + 1, size2=size * 5 + 4)
                head2 = ("|" + "{{:^{size:d}s}}|" * 12).format(size=size)
                line = ("|{{:^{size:d}d}}|" + "{{:^{size:d}.4f}}|" * 11).format(size=size)
                logging.info(head1.format("", "train", "validation"))
                logging.info(head2.format("epoch", "time", *(("loss", "diff_loss", "diff", "q_loss", "q") * 2)))
                for epoch in range(1, MÜNCHHAUSEN_EPOCHS + 1):
                    clock = time.time()
                    train_loss, *_ = self.epoch(session, self.get_train_set(), self.get_optimiser())
                    validation_loss, *_ = self.epoch(session, self.get_validation_set())
                    delay = time.time() - clock
                    logging.info(line.format(epoch, delay, *train_loss, *validation_loss))
                    figure.plot(epoch, train_loss[0], ".b")
                    figure.plot(epoch, validation_loss[0], ".r")
                    self.save(session)

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
