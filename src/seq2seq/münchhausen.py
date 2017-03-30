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
        super().__init__("münchhausen")
        self.indexes = None
        self.inputs = None
        self.inputs_sizes = None
        self.output = None
        self.evaluation = None
        self.optimise = None
        self.q = None
        self.losses = None

    @trace
    def build_inputs(self):
        indexes = {}
        inputs_sizes = {}
        for label in PARTS:
            with vs.variable_scope(label):
                indexes[label] = []
                for i in range(INPUT_SIZE):
                    placeholder = tf.placeholder(tf.int32, [BATCH_SIZE], "indexes")
                    indexes[label].append(placeholder)
                inputs_sizes[label] = tf.placeholder(tf.int32, [BATCH_SIZE], "input_sizes")
        evaluation = tf.placeholder(tf.float32, [BATCH_SIZE], "evaluation")
        embeddings = tf.constant(np.asarray(Embeddings.embeddings()))
        inputs = {label: [tf.gather(embeddings, inp) for inp in indexes[label]] for label in PARTS}
        self.indexes = indexes
        self.inputs_sizes = inputs_sizes
        self.evaluation = evaluation
        self.inputs = inputs

    @trace
    def build_outputs(self):
        inputs = self.inputs
        inputs_sizes = self.inputs_sizes
        with vs.variable_scope("munchhausen"):
            inputs_states = []
            for label in PARTS:
                with vs.variable_scope(label):
                    fw, bw = build_encoder(inputs[label], INPUT_STATE_SIZE, inputs_sizes[label])
                    inputs_states.append(tf.concat(axis=1, values=[fw[0], bw[-1]]))
                    inputs_states.append(tf.concat(axis=1, values=[fw[-1], bw[0]]))
            goes = tf.stack([Embeddings.get_embedding(GO) for _ in range(BATCH_SIZE)])
            inputs_states = tf.transpose(tf.stack(inputs_states), [1, 0, 2])
            with vs.variable_scope("output"):
                decoder_inputs = [goes for _ in range(OUTPUT_SIZE)]
                output, _ = build_decoder(decoder_inputs, inputs_states, INPUT_STATE_SIZE, INITIAL_STATE,
                                          EMBEDDING_SIZE, loop=True)
            with vs.variable_scope("evaluation"):
                fw, bw = build_encoder(output, OUTPUT_STATE_SIZE)
                output_states = [tf.concat(axis=1, values=[fw[i], bw[-(i + 1)]]) for i in range(OUTPUT_SIZE)]
                Q, _ = build_decoder(output_states, inputs_states, INPUT_STATE_SIZE, INITIAL_STATE, 1)
                Q = tf.transpose(tf.reshape(tf.stack(tf.nn.relu(Q)), [OUTPUT_SIZE, BATCH_SIZE]), [1, 0])
                W_shape = [OUTPUT_STATE_SIZE * 2, 1]
                B_shape = [1]
                W = tf.Variable(initial_value=tf.truncated_normal(W_shape, dtype=tf.float32), name="weights")
                B = tf.Variable(initial_value=tf.truncated_normal(B_shape, dtype=tf.float32), name="biases")
                output_states = tf.reshape(tf.stack(output_states), [BATCH_SIZE * OUTPUT_SIZE, OUTPUT_STATE_SIZE * 2])
                I = tf.sigmoid(tf.reshape(tf.matmul(output_states, W) + B, [BATCH_SIZE, OUTPUT_SIZE]))
                q = tf.reduce_sum(Q * I, axis=1)
            with vs.variable_scope("softmax"):
                W_shape = [EMBEDDING_SIZE, NUM_TOKENS]
                B_shape = [NUM_TOKENS]
                W = tf.Variable(initial_value=tf.truncated_normal(W_shape, dtype=tf.float32), name="weights")
                B = tf.Variable(initial_value=tf.truncated_normal(B_shape, dtype=tf.float32), name="biases")
                output = tf.reshape(tf.stack(output), [BATCH_SIZE * OUTPUT_SIZE, EMBEDDING_SIZE])
                logits = tf.reshape(tf.matmul(output, W) + B, [OUTPUT_SIZE, BATCH_SIZE, NUM_TOKENS])
                output = tf.nn.softmax(logits)
                output = tf.unstack(output)
            scope = vs.get_variable_scope().name
        self.output = output
        self.q = q
        self.scope = scope

    @trace
    def build_fetches(self):
        with vs.variable_scope("loss"):
            trainable_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope)
            regularization_variables = [self.scope + "/" + variable for variable in REGULARIZATION_VARIABLES]
            diff = DIFF_WEIGHT * tf.reduce_mean(tf.square(self.evaluation - self.q))
            l2_loss = L2_WEIGHT * build_l2_loss(trainable_variables, regularization_variables)
            q = Q_WEIGHT * tf.reduce_mean(self.q)
            loss = diff + l2_loss + q
        with vs.variable_scope("optimiser"):
            optimise = tf.train.AdamOptimizer(beta1=0.85).minimize(loss, var_list=trainable_variables)
        self.optimise = optimise
        self.losses = (loss, diff, l2_loss, q)

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
        # fixme: ondin pidoras
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

    def epoch(self, session, data_set, optimise=True):
        losses = []
        for indexes, inputs_sizes in data_set:
            feed_dict = {}
            for label in PARTS:
                feed_dict.update(zip(self.indexes[label], indexes[label]))
                feed_dict[self.inputs_sizes[label]] = inputs_sizes[label]
            fetches = (self.output,)
            output, *_ = session.run(fetches=fetches, feed_dict=feed_dict)
            feed_dict[self.evaluation] = contracts.evaluate(indexes, output)
            fetches = (self.losses, self.optimise) if optimise else (self.losses,)
            local_losses, *_ = session.run(fetches=fetches, feed_dict=feed_dict)
            losses.append(local_losses)
        return np.mean(losses, axis=(0,))

    @trace
    def train(self, restore: bool = False, epochs=Q_FUNCTION_EPOCHS):
        with tf.Session() as session, tf.device('/cpu:0'), Figure(xauto=True) as figure:
            session.run(tf.global_variables_initializer())
            if restore:
                self.restore(session)
            size = 13
            head1 = ("|{{:^{size1:d}s}}|" + "{{:^{size2:d}s}}|" * 2).format(size1=size * 2 + 1, size2=size * 4 + 3)
            head2 = ("|" + "{{:^{size:d}s}}|" * 10).format(size=size)
            line = ("|{{:^{size:d}d}}|" + "{{:^{size:d}.4f}}|" * 9).format(size=size)
            logging.info(head1.format("", "train", "validation"))
            logging.info(head2.format("epoch", "time", *(("loss", "diff", "l2_loss", "q") * 2)))
            for epoch in range(1, epochs + 1):
                clock = time.time()
                train_loss = self.epoch(session, self.get_train_set())
                validation_loss = self.epoch(session, self.get_validation_set(), False)
                delay = time.time() - clock
                logging.info(line.format(epoch, delay, *train_loss, *validation_loss))
                figure.plot(epoch, train_loss[0], ".b")
                figure.plot(epoch, validation_loss[0], ".r")
                self.save(session)

    @trace
    def test(self):
        pass

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
