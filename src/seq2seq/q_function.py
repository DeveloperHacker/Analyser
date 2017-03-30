from multiprocessing import Pool

from seq2seq import contracts
from seq2seq.Net import *
from seq2seq.seq2seq import *
from utils import batcher, dumper
from utils.Figure import Figure
from utils.handlers import SIGINTException
from utils.wrapper import *
from variables.embeddings import *
from variables.path import *
from variables.sintax import *
from variables.tags import *
from variables.train import *


class QFunctionNet(Net):
    def __init__(self):
        super().__init__("q-function")
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
        with vs.variable_scope("output"):
            output = []
            for i in range(OUTPUT_SIZE):
                placeholder = tf.placeholder(tf.float32, [BATCH_SIZE, NUM_TOKENS], "output")
                output.append(placeholder)
        evaluation = tf.placeholder(tf.float32, [BATCH_SIZE], "evaluation")
        self.indexes = indexes
        self.inputs_sizes = inputs_sizes
        self.output = output
        self.evaluation = evaluation

    @trace
    def substitute_embeddings(self):
        indexes = self.indexes
        embeddings = tf.constant(np.asarray(Embeddings.embeddings()))
        self.inputs = {label: [tf.gather(embeddings, inp) for inp in indexes[label]] for label in PARTS}

    @trace
    def build_outputs(self):
        inputs = self.inputs
        inputs_sizes = self.inputs_sizes
        output = self.output
        with vs.variable_scope("q-function"):
            inputs_states = []
            for label in PARTS:
                with vs.variable_scope(label):
                    states_fw, states_bw = build_encoder(inputs[label], INPUT_STATE_SIZE, inputs_sizes[label])
                    inputs_states.append(tf.concat(axis=1, values=[states_fw[0], states_bw[-1]]))
                    inputs_states.append(tf.concat(axis=1, values=[states_fw[-1], states_bw[0]]))
            with vs.variable_scope("output"):
                states_fw, states_bw = build_encoder(output, OUTPUT_STATE_SIZE)
                output_states = [tf.concat(axis=1, values=[states_fw[i], states_bw[-(i + 1)]]) for i in
                                 range(OUTPUT_SIZE)]
            inputs_states = tf.transpose(tf.stack(inputs_states), [1, 0, 2])
            Q, _ = build_decoder(output_states, inputs_states, INPUT_STATE_SIZE, INITIAL_STATE, 1)
            Q = tf.nn.relu(Q)
            Q = tf.transpose(tf.reshape(tf.stack(Q), [OUTPUT_SIZE, BATCH_SIZE]), [1, 0])
            with vs.variable_scope("evaluation"):
                W_shape = [OUTPUT_STATE_SIZE * 2, 1]
                B_shape = [1]
                W = tf.Variable(initial_value=tf.truncated_normal(W_shape, dtype=tf.float32), name="weights")
                B = tf.Variable(initial_value=tf.truncated_normal(B_shape, dtype=tf.float32), name="biases")
                output_states = tf.reshape(tf.stack(output_states), [BATCH_SIZE * OUTPUT_SIZE, OUTPUT_STATE_SIZE * 2])
                I = tf.sigmoid(tf.reshape(tf.matmul(output_states, W) + B, [BATCH_SIZE, OUTPUT_SIZE]))
                q = tf.reduce_sum(Q * I, axis=1)
            scope = vs.get_variable_scope().name
        self.q = q
        self.scope = scope

    @trace
    def build_fetches(self):
        with vs.variable_scope("loss"):
            trainable_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope)
            regularization_variables = [self.scope + "/" + variable for variable in REGULARIZATION_VARIABLES]
            diff = tf.reduce_mean(tf.square(self.evaluation - self.q))
            l2_loss = build_l2_loss(trainable_variables, regularization_variables)
            loss = DIFF_WEIGHT * diff + L2_WEIGHT * l2_loss
        with vs.variable_scope("optimiser"):
            optimise = tf.train.AdamOptimizer(beta1=0.90).minimize(loss, var_list=trainable_variables)
        self.optimise = optimise
        self.losses = (loss, diff, l2_loss)

    @staticmethod
    def most_different(samples: list, most: int):
        NUM_COLUMNS = 4

        minimum = min(samples, key=lambda x: x[1])[1]
        maximum = max(samples, key=lambda x: x[1])[1]
        minimum -= minimum * 1e-4
        maximum += maximum * 1e-4
        delta = (maximum - minimum) / NUM_COLUMNS
        baskets = batcher.hist(samples, list(np.arange(minimum, maximum, delta))[1:], key=lambda x: x[1])
        maximum = len(max(baskets.items(), key=lambda x: len(x[1]))[1])
        line = None
        for i in range(maximum):
            num_samples = sum([min(len(samples), i) for _, samples in baskets.items()])
            if num_samples >= most:
                line = i
                break
        assert line is not None, "NS: {} < M: {}".format(len(samples), most)
        samples = []
        for _, basket in baskets.items():
            random.shuffle(basket)
            samples.extend(basket[:min(len(basket), line)])
        return samples[:most]

    @staticmethod
    def build_samples(doc, evaluate):
        ARGUMENT = 1
        FUNCTION = 2

        NUM_SAMPLES = 100
        NUM_TRUE_SAMPLES = 20

        indexes = {}
        inputs_sizes = {}
        for label, embeddings in doc:
            indexes[label] = []
            inputs_sizes[label] = []
            for _ in range(INPUT_SIZE):
                indexes[label].append([])
            line = embeddings + [Embeddings.get_index(PAD) for _ in range(INPUT_SIZE - len(embeddings))]
            inputs_sizes[label].append(len(embeddings))
            for i, embedding in enumerate(line):
                indexes[label][i].append(embedding)
        samples = []
        for _ in range(NUM_SAMPLES):
            sample = []
            state = FUNCTION
            expected = OUTPUT_SIZE
            arguments = None
            num_functions = random.randrange(OUTPUT_SIZE // 2)
            while True:
                if state == FUNCTION:
                    i = random.randrange(len(Functions))
                    function = Functions[i]
                    arguments = function.arguments
                    expected -= arguments + 1
                    if expected <= 0 or num_functions == 0:
                        sample.append(Tokens.END.embedding)
                        sample.extend([Tokens.NOP.embedding] * (arguments + expected))
                        break
                    sample.append(function.embedding)
                    num_functions -= 1
                    state = ARGUMENT
                elif state == ARGUMENT:
                    i = random.randrange(len(Constants))
                    constant = Constants[i]
                    sample.append(constant.embedding)
                    arguments -= 1
                    if arguments == 0:
                        state = FUNCTION
            samples.append((sample, evaluate(indexes, np.expand_dims(sample, axis=1))[0]))
        true_samples = QFunctionNet.most_different(samples, NUM_TRUE_SAMPLES)
        random.shuffle(samples)
        noised_samples = QFunctionNet.noise_samples(samples[:NUM_SAMPLES - NUM_TRUE_SAMPLES])
        return (indexes, inputs_sizes), true_samples, noised_samples

    @staticmethod
    @trace
    def noise_samples(inputs, samples, evaluate):
        NUM_GENETIC_CYCLES = 10
        NOISE_DEPTH = 3

        noised_samples = list(samples)
        num_samples = len(noised_samples)
        for i in range(NUM_GENETIC_CYCLES):
            for j in range(num_samples):
                sample = noised_samples[j][0][::]
                indexes = random.sample(list(np.arange(len(sample))), NOISE_DEPTH)
                for index in indexes:
                    n = random.randrange(NUM_TOKENS)
                    sample[index] = Tokens.get(n).embedding
                noised_samples.append((sample, evaluate(inputs, np.expand_dims(sample, axis=1))[0]))
            noised_samples = QFunctionNet.most_different(noised_samples, num_samples)
        return noised_samples

    @staticmethod
    def build_samples_wrapper(doc, evaluate):
        inputs, true_samples, noised_samples = QFunctionNet.build_samples(doc, evaluate)
        return [(inputs, sample) for sample in true_samples + noised_samples]

    @staticmethod
    def build_batch(batch: list):
        indexes = {label: [] for label in PARTS}
        inputs_sizes = {label: [] for label in PARTS}
        samples = []
        evaluations = []
        for (inp, inp_sizes), (sample, evaluation) in batch:
            for label in PARTS:
                indexes[label].append(inp[label])
                inputs_sizes[label].append(inp_sizes[label])
            samples.append(sample)
            evaluations.append(evaluation)
        for label in PARTS:
            indexes[label] = np.transpose(np.asarray(indexes[label]), axes=(1, 0, 2))
            indexes[label] = np.squeeze(indexes[label], axis=(2,))
            inputs_sizes[label] = np.squeeze(np.asarray(inputs_sizes[label]), axis=(1,))
        samples = np.transpose(np.asarray(samples), axes=(1, 0, 2))
        return indexes, inputs_sizes, samples, evaluations

    @staticmethod
    def indexes(method):
        doc = []
        for label, (embeddings, text) in method:
            embeddings = [Embeddings.get_index(word) for embedding, word in zip(embeddings, text)]
            doc.append((label, embeddings))
        return doc

    @staticmethod
    @trace
    def build_data_set():
        evaluate = contracts.evaluate
        methods = dumper.load(VEC_METHODS)
        with Pool() as pool:
            docs = pool.map(QFunctionNet.indexes, methods)
            docs_baskets = batcher.throwing(docs, [INPUT_SIZE])
            docs = docs_baskets[INPUT_SIZE]
            raw_samples = pool.starmap(QFunctionNet.build_samples_wrapper, ((doc, evaluate) for doc in docs))
            samples = [sample for samples in raw_samples for sample in samples]
            random.shuffle(samples)
            batches = batcher.chunks(samples, BATCH_SIZE)
            batches = pool.map(QFunctionNet.build_batch, batches)
        dumper.dump(batches, Q_FUNCTION_BATCHES)

    @staticmethod
    @trace
    def data_set() -> list:
        return dumper.load(Q_FUNCTION_BATCHES)

    @trace
    def build_feed_dicts(self, batches):
        indexes = self.indexes
        inputs_sizes = self.inputs_sizes
        output = self.output
        evaluation = self.evaluation
        feed_dicts = []
        for _indexes, _inputs_sizes, _samples, _evaluations in batches:
            feed_dict = {}
            feed_dict.update(zip(output, _samples))
            for label in PARTS:
                feed_dict.update(zip(indexes[label], _indexes[label]))
                feed_dict[inputs_sizes[label]] = _inputs_sizes[label]
            feed_dict[evaluation] = _evaluations
            feed_dicts.append(feed_dict)
        random.shuffle(feed_dicts)
        train_set = feed_dicts[:int(len(feed_dicts) * TRAIN_SET)]
        validation_set = feed_dicts[len(train_set):]
        self.train_set = train_set
        self.validation_set = validation_set

    @trace
    def build(self):
        self.build_inputs()
        self.substitute_embeddings()
        self.build_outputs()
        self.build_fetches()
        self.build_feed_dicts(QFunctionNet.data_set())

    @trace
    def train(self, restore: bool = False, epochs=Q_FUNCTION_EPOCHS):
        fetches = (
            self.optimise,
            self.losses
        )
        fail = True
        fail_counter = 0
        while fail:
            with tf.Session() as session, tf.device('/cpu:0'), Figure(xauto=True) as figure:
                try:
                    session.run(tf.global_variables_initializer())
                    if restore:
                        self.restore(session)
                    size = 13
                    head1 = ("|" + "{{:^{size1:d}s}}|" + "{{:^{size2:d}s}}|" * 2).format(
                        size1=size * 2 + 1,
                        size2=size * 3 + 2)
                    head2 = ("|" + "{{:^{size:d}s}}|" * 8).format(size=size)
                    line = ("|" + "{{:^{size:d}.4f}}|" * 8).format(size=size)
                    logging.info(head1.format("", "train", "validation"))
                    logging.info(head2.format("epoch", "time", "loss", "diff", "l2_loss", "loss", "diff", "l2_loss"))
                    for epoch in range(1, epochs + 1):
                        train_losses = []
                        clock = time.time()
                        for feed_dict in self.get_train_set():
                            _, losses = session.run(fetches=fetches, feed_dict=feed_dict)
                            train_losses.append(losses)
                        validation_losses = []
                        for feed_dict in self.get_validation_set():
                            losses = session.run(fetches=self.losses, feed_dict=feed_dict)
                            validation_losses.append(losses)
                        delay = time.time() - clock
                        train_loss = np.mean(train_losses, axis=(0,))
                        validation_loss = np.mean(validation_losses, axis=(0,))
                        logging.info(line.format(epoch, delay, *train_loss, *validation_loss))
                        figure.plot(epoch, train_loss[0], ".b")
                        figure.plot(epoch, validation_loss[0], ".r")
                        self.save(session)
                        fail = epoch in LPR and (train_loss[0] > LPR[epoch] or validation_loss[0] > LPR[epoch])
                        if fail:
                            self.mkdir()
                            logging.info("FAIL")
                            fail_counter += 1
                            break
                except SIGINTException:
                    logging.info("SIGINT")
                finally:
                    self.save(session)
        print("Number of fails: {}".format(fail_counter))

    @trace
    def test(self):
        pass

    @staticmethod
    @trace
    def run(foo: str):
        q_function_net = QFunctionNet()
        q_function_net.build()
        if foo == "train":
            q_function_net.train()
        elif foo == "restore":
            q_function_net.train(True)
        elif foo == "test":
            q_function_net.test()
