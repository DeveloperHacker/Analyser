from seq2seq import contracts
from seq2seq.Net import *
from seq2seq.q_function import QFunctionNet
from seq2seq.seq2seq import *
from utils import batcher, dumper
from utils.Figure import Figure
from utils.handlers import SIGINTException
from utils.wrapper import *
from variables.embeddings import *
from variables.path import *
from variables.sintax import NUM_TOKENS, Tokens
from variables.tags import *
from variables.train import *


class AnalyserNet(Net):
    def __init__(self):
        super().__init__("analyser")
        self.indexes = None
        self.inputs = None
        self.inputs_sizes = None
        self.output = None
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
                    placeholder = tf.placeholder(tf.float32, [BATCH_SIZE], "indexes")
                    indexes[label].append(placeholder)
                inputs_sizes[label] = tf.placeholder(tf.int32, [BATCH_SIZE], "input_sizes")
        self.indexes = indexes
        self.inputs_sizes = inputs_sizes

    @trace
    def substitute_embeddings(self):
        indexes = self.indexes
        embeddings = tf.constant(np.asarray(Embeddings.embeddings()))
        self.inputs = {label: [tf.gather(embeddings, inp) for inp in indexes[label]] for label in PARTS}

    @trace
    def build_outputs(self):
        inputs = self.inputs
        inputs_sizes = self.inputs_sizes
        with vs.variable_scope("analyser"):
            inputs_states = []
            for label in PARTS:
                with vs.variable_scope(label):
                    output_states_fw, output_states_bw = build_encoder(inputs[label], INPUT_STATE_SIZE,
                                                                       inputs_sizes[label])
                    inputs_states.append(tf.concat(axis=1, values=[output_states_fw[0], output_states_bw[-1]]))
                    inputs_states.append(tf.concat(axis=1, values=[output_states_fw[-1], output_states_bw[0]]))
            goes = tf.stack([Embeddings.get_embedding(GO) for _ in range(BATCH_SIZE)])
            decoder_inputs = [goes for _ in range(OUTPUT_SIZE)]
            inputs_states = tf.stack(inputs_states)
            inputs_states = tf.transpose(inputs_states, [1, 0, 2])
            output, _ = build_decoder(decoder_inputs, inputs_states, INPUT_STATE_SIZE, INITIAL_STATE, EMBEDDING_SIZE,
                                      loop=True)
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
        self.scope = scope

    @trace
    def build_fetches(self):
        with vs.variable_scope("loss"):
            trainable_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope)
            regularization_variables = [self.scope + "/" + variable for variable in REGULARIZATION_VARIABLES]
            l2_loss = build_l2_loss(trainable_variables, regularization_variables)
            q = tf.reduce_mean(self.q)
            _, oh_loss = tf.nn.moments(tf.stack(self.output), axes=(1,))
            oh_loss = tf.log(2.0 - tf.reduce_mean(oh_loss))
            loss = Q_WEIGHT * q + L2_WEIGHT * l2_loss
        with vs.variable_scope("optimiser"):
            optimiser = tf.train.AdamOptimizer(beta1=0.90).minimize(loss, var_list=trainable_variables)
        self.optimisers = (optimiser,)
        self.losses = (loss, q, l2_loss, oh_loss)

    @staticmethod
    @trace
    def build_data_set():
        methods = dumper.load(VEC_METHODS)
        docs = [QFunctionNet.indexes(method) for method in methods]
        docs_baskets = batcher.throwing(docs, [INPUT_SIZE])
        docs = docs_baskets[INPUT_SIZE]
        batches = batcher.build_batches(docs, BATCH_SIZE)
        dumper.dump(batches, ANALYSER_BATCHES)

    @staticmethod
    @trace
    def data_set():
        raw_batches = dumper.load(ANALYSER_BATCHES)
        batches = []
        for batch in raw_batches:
            indexes = {label: [] for label in PARTS}
            inputs_sizes = {label: [] for label in PARTS}
            for label, lines in batch:
                inputs_sizes[label] = []
                for embeddings in lines:
                    line = embeddings + [Embeddings.get_index(PAD) for _ in range(INPUT_SIZE - len(embeddings))]
                    indexes[label].append(line)
                    inputs_sizes[label].append(len(embeddings))
                indexes[label] = np.transpose(np.asarray(indexes[label]), axes=(1, 0))
            batches.append((indexes, inputs_sizes))
        return batches

    @trace
    def build_feed_dicts(self, batches):
        indexes = self.indexes
        inputs_sizes = self.inputs_sizes
        feed_dicts = []
        for _indexes, _inputs_sizes in batches:
            feed_dict = {}
            for label in PARTS:
                feed_dict.update(zip(indexes[label], _indexes[label]))
                feed_dict[inputs_sizes[label]] = _inputs_sizes[label]
            feed_dicts.append(feed_dict)
        train_set = feed_dicts[:int(len(feed_dicts) * TRAIN_SET)]
        validation_set = feed_dicts[len(train_set):]
        self.train_set = train_set
        self.validation_set = validation_set

    @trace
    def build(self) -> QFunctionNet:
        q_function_net = QFunctionNet()
        q_function_net.build_inputs()
        q_function_net.substitute_embeddings()
        self.indexes = q_function_net.indexes
        self.inputs_sizes = q_function_net.inputs_sizes
        self.substitute_embeddings()
        self.build_outputs()
        q_function_net.output = self.output
        q_function_net.build_outputs()
        self.q = q_function_net.q
        self.build_fetches()
        self.build_feed_dicts(AnalyserNet.data_set())
        return q_function_net

    @trace
    def pretrain(self):
        with tf.Session() as session:
            session.run(tf.global_variables_initializer())
            self.save(session)

    @trace
    def train(self, q_function_net: QFunctionNet, restore: bool = False):
        data = None
        validation_fetches = (
            self.indexes,
            self.inputs,
            self.inputs_sizes,
            self.output,
            self.losses
        )
        train_fetches = (self.get_optimiser(),) + validation_fetches
        with tf.Session() as session, tf.device('/cpu:0'), Figure(xauto=True) as figure:
            try:
                session.run(tf.global_variables_initializer())
                q_function_net.restore(session)
                if restore:
                    self.restore(session)
                size = 13
                head1 = ("|" + "{{:^{size1:d}s}}|" + "{{:^{size2:d}s}}|" * 2).format(
                    size1=size // 2 + 1 + size + 1,
                    size2=size * 5 + 4)
                head2 = ("|" + "{{:^{size1:d}s}}|" + "{{:^{size:d}s}}|" * 11).format(size1=size // 2 + 1, size=size)
                line = ("|" + "{{:^{size1:d}d}}|" + "{{:^{size:d}.4f}}|" * 11).format(size1=size // 2 + 1, size=size)
                logging.info(head1.format("", "train", "validation"))
                logging.info(head2.format("epoch", "time", *(("loss", "q", "l2_loss", "oh_loss", "eval") * 2)))
                for epoch in range(1, ANALYSER_EPOCHS + 1):
                    data = []
                    train_losses = []
                    validation_losses = []
                    clock = time.time()
                    for feed_dict in self.get_train_set():
                        _, indexes, inputs, inputs_sizes, output, losses = session.run(fetches=train_fetches,
                                                                                       feed_dict=feed_dict)
                        evaluation = contracts.evaluate(inputs, output)
                        data.append((indexes, inputs_sizes, output, evaluation))
                        train_losses.append(losses + (np.mean(evaluation),))
                    for feed_dict in self.get_validation_set():
                        indexes, inputs, inputs_sizes, output, losses = session.run(fetches=validation_fetches,
                                                                                    feed_dict=feed_dict)
                        evaluation = contracts.evaluate(inputs, output)
                        data.append((indexes, inputs_sizes, output, evaluation))
                        validation_losses.append(losses + (np.mean(evaluation),))
                    delay = time.time() - clock
                    train_loss = np.mean(train_losses, axis=(0,))
                    validation_loss = np.mean(validation_losses, axis=(0,))
                    logging.info(line.format(epoch, delay, *train_loss, *validation_loss))
                    figure.plot(epoch, train_loss[0], ".b")
                    figure.plot(epoch, validation_loss[0], ".r")
                    self.save(session)
            except SIGINTException:
                logging.info("SIGINT")
            finally:
                self.save(session)
        return data

    @trace
    def test(self, q_function_net: QFunctionNet):
        fetches = (
            self.inputs,
            self.output,
            self.losses
        )
        with tf.Session() as session:
            session.run(tf.global_variables_initializer())
            q_function_net.restore(session)
            self.restore(session)
            for feed_dict in self.train_set:
                _inputs, _output, (loss, *_) = session.run(fetches=fetches, feed_dict=feed_dict)
                print(_output)
                output = [[] for _ in range(len(_output))]
                for i, distributions in enumerate(_output):
                    indexes = np.argmax(distributions, axis=1)
                    for idx in indexes:
                        output[i].append(Tokens.get(idx).name)
                evaluation = np.mean(contracts.evaluate(_inputs, _output))
                for tokens in output:
                    print(" ".join(("{:10s}".format(token) for token in tokens)), loss, evaluation)

    @staticmethod
    @trace
    def run(foo: str):
        analyser_net = AnalyserNet()
        q_function_net = analyser_net.build()
        if foo == "pretrain":
            analyser_net.pretrain()
        elif foo == "train":
            analyser_net.train(q_function_net)
        elif foo == "restore":
            analyser_net.train(q_function_net, True)
        elif foo == "test":
            analyser_net.test(q_function_net)
