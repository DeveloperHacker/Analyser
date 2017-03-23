from multiprocessing import Pool

from seq2seq import contract
from seq2seq.Net import *
from seq2seq.seq2seq import *
from utils import batcher, dumper
from utils.Figure import Figure
from utils.handlers import SIGINTException
from utils.wrapper import *
from variables.embeddings import INITIAL_STATE, PAD
from variables.path import SEQ2SEQ, VEC_METHODS
from variables.sintax import *
from variables.tags import *
from variables.train import *


@trace
def build_inputs():
    inputs = {}
    inputs_sizes = {}
    for label in PARTS:
        with vs.variable_scope(label):
            inputs[label] = []
            for i in range(INPUT_SIZE):
                placeholder = tf.placeholder(tf.float32, [BATCH_SIZE, EMBEDDING_SIZE], "input")
                inputs[label].append(placeholder)
            inputs_sizes[label] = tf.placeholder(tf.int32, [BATCH_SIZE], "input_sizes")
    with vs.variable_scope("output"):
        output = []
        for i in range(OUTPUT_SIZE):
            placeholder = tf.placeholder(tf.float32, [BATCH_SIZE, NUM_TOKENS], "output")
            output.append(placeholder)
    evaluation = tf.placeholder(tf.float32, [BATCH_SIZE], "evaluation")
    return inputs, inputs_sizes, output, evaluation


@trace
def build_outputs(inputs, inputs_sizes, output):
    with vs.variable_scope("q-function"):
        inputs_states = []
        for label in PARTS:
            with vs.variable_scope(label):
                states_fw, states_bw = build_encoder(inputs[label], INPUT_STATE_SIZE, inputs_sizes[label])
                inputs_states.append(tf.concat(axis=1, values=[states_fw[0], states_bw[-1]]))
                inputs_states.append(tf.concat(axis=1, values=[states_fw[-1], states_bw[0]]))
        with vs.variable_scope("output"):
            states_fw, states_bw = build_encoder(output, OUTPUT_STATE_SIZE)
            output_states = [tf.concat(axis=1, values=[states_fw[i], states_bw[-(i + 1)]]) for i in range(OUTPUT_SIZE)]
        inputs_states = tf.transpose(tf.stack(inputs_states), [1, 0, 2])
        Q, _ = build_decoder(output_states, inputs_states, INPUT_STATE_SIZE, INITIAL_STATE, 1)
        Q = tf.nn.relu(Q)
        Q = tf.transpose(tf.reshape(tf.stack(Q), [OUTPUT_SIZE, BATCH_SIZE]), [1, 0])
        with vs.variable_scope("loss"):
            W_shape = [OUTPUT_STATE_SIZE * 2, 1]
            B_shape = [1]
            std = INITIALIZATION_STD
            W = tf.Variable(initial_value=tf.truncated_normal(W_shape, stddev=std, dtype=tf.float32), name="weights")
            B = tf.Variable(initial_value=tf.truncated_normal(B_shape, stddev=std, dtype=tf.float32), name="biases")
            output_states = tf.reshape(tf.stack(output_states), [BATCH_SIZE * OUTPUT_SIZE, OUTPUT_STATE_SIZE * 2])
            I = tf.sigmoid(tf.reshape(tf.matmul(output_states, W) + B, [BATCH_SIZE, OUTPUT_SIZE]))
            q = tf.reduce_sum(Q * I, axis=1)
        scope = vs.get_variable_scope().name
    return q, scope


@trace
def build_fetches(q_function_scope, evaluation, q):
    with vs.variable_scope("loss"):
        trainable_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, q_function_scope)
        loss = tf.reduce_mean(tf.square(evaluation - q))
    with vs.variable_scope("optimiser"):
        optimise = tf.train.AdamOptimizer(beta1=0.90).minimize(loss, var_list=trainable_variables)
    return optimise, (loss,)


def most_different(samples: list, most: int, num_columns: int):
    minimum = min(samples, key=lambda x: x[1])[1]
    maximum = max(samples, key=lambda x: x[1])[1]
    minimum -= minimum * 1e-4
    maximum += maximum * 1e-4
    delta = (maximum - minimum) / num_columns
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


def build_samples(doc, evaluate):
    ARGUMENT = 1
    FUNCTION = 2

    NUM_SAMPLES = 20
    NUM_TRUE_SAMPLES = 7
    NUM_GENETIC_CYCLES = 10
    NOISE_DEPTH = 3
    NUM_COLUMNS = 4

    inputs = {}
    inputs_sizes = {}
    for label, (embeddings, _) in doc:
        inputs[label] = []
        inputs_sizes[label] = []
        for _ in range(INPUT_SIZE):
            inputs[label].append([])
        line = embeddings + [PAD for _ in range(INPUT_SIZE - len(embeddings))]
        inputs_sizes[label].append(len(embeddings))
        for i, embedding in enumerate(line):
            inputs[label][i].append(embedding)
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
        samples.append((sample, evaluate(inputs, np.expand_dims(sample, axis=1))[0]))
    true_samples = most_different(samples, NUM_TRUE_SAMPLES, NUM_COLUMNS)
    random.shuffle(samples)
    noised_samples = samples[:NUM_SAMPLES - NUM_TRUE_SAMPLES]
    for i in range(NUM_GENETIC_CYCLES):
        for j in range(NUM_SAMPLES - NUM_TRUE_SAMPLES):
            sample = noised_samples[j][0][::]
            indexes = random.sample(list(np.arange(len(sample))), NOISE_DEPTH)
            for index in indexes:
                n = random.randrange(NUM_TOKENS)
                sample[index] = Tokens.get(n).embedding
            noised_samples.append((sample, evaluate(inputs, np.expand_dims(sample, axis=1))[0]))
        noised_samples = most_different(noised_samples, NUM_SAMPLES - NUM_TRUE_SAMPLES, NUM_COLUMNS)
    return (inputs, inputs_sizes), true_samples, noised_samples


def build_samples_wrapper(doc, evaluate):
    inputs, true_samples, noised_samples = build_samples(doc, evaluate)
    return [(inputs, sample) for sample in true_samples + noised_samples]


def build_batch(batch: list):
    inputs = {}
    inputs_sizes = {}
    for label in PARTS:
        inputs[label] = []
        inputs_sizes[label] = []
    samples = []
    evaluations = []
    for (inp, inp_sizes), (sample, evaluation) in batch:
        for label in PARTS:
            inputs[label].append(inp[label])
            inputs_sizes[label].append(inp_sizes[label])
        samples.append(sample)
        evaluations.append(evaluation)
    for label in PARTS:
        inputs[label] = np.transpose(inputs[label], axes=(1, 0, 3, 2))
        inputs[label] = np.squeeze(inputs[label], axis=(3,))
        inputs_sizes[label] = np.squeeze(inputs_sizes[label], axis=(1,))
    samples = np.transpose(samples, axes=(1, 0, 2))
    return inputs, inputs_sizes, samples, evaluations


@trace
def build_data_set(evaluate):
    methods = dumper.load(VEC_METHODS)
    with Pool() as pool:
        methods_baskets = batcher.throwing(methods, [INPUT_SIZE])
        docs = methods_baskets[INPUT_SIZE]
        raw_samples = pool.starmap(build_samples_wrapper, ((doc, evaluate) for doc in docs))
        samples = [sample for samples in raw_samples for sample in samples]
        random.shuffle(samples)
        batches = batcher.chunks(samples, BATCH_SIZE)
        batches = pool.map(build_batch, batches)
    return batches


@trace
def build_feed_dicts(inputs, inputs_sizes, output, evaluation):
    batches = build_data_set(contract.evaluate)
    feed_dicts = []
    for _inputs, _inputs_sizes, _samples, _evaluations in batches:
        feed_dict = {}
        feed_dict.update(zip(output, _samples))
        for label in PARTS:
            feed_dict.update(zip(inputs[label], _inputs[label]))
            feed_dict[inputs_sizes[label]] = _inputs_sizes[label]
        feed_dict[evaluation] = _evaluations
        feed_dicts.append(feed_dict)
    random.shuffle(feed_dicts)
    train_set = feed_dicts[:int(len(feed_dicts) * TRAIN_SET)]
    validation_set = feed_dicts[len(train_set):]
    return train_set, validation_set


@trace
def build() -> QFunctionNet:
    q_function_net = QFunctionNet()

    inputs, inputs_sizes, output, evaluation = build_inputs()
    q_function_net.inputs = inputs
    q_function_net.inputs_sizes = inputs_sizes
    q_function_net.output = output
    q_function_net.evaluation = evaluation

    q, q_function_scope = build_outputs(inputs, inputs_sizes, output)
    q_function_net.q = q
    q_function_net.scope = q_function_scope

    train_set, validation_set = build_feed_dicts(inputs, inputs_sizes, output, evaluation)
    q_function_net.train_set = train_set
    q_function_net.validation_set = validation_set

    optimise, losses = build_fetches(q_function_scope, evaluation, q)
    q_function_net.optimise = optimise
    q_function_net.losses = losses
    return q_function_net


@trace
def train(q_function_net: QFunctionNet, restore: bool = False, epochs=Q_FUNCTION_EPOCHS):
    fetches = (
        q_function_net.optimise,
        q_function_net.losses
    )
    with tf.Session() as session, tf.device('/cpu:0'), Figure(xauto=True) as figure:
        try:
            session.run(tf.global_variables_initializer())
            if restore:
                q_function_net.restore(session)
            raw_size = 10
            formatter = "{{:^{size:d}s}}{{:^{size:d}s}}{{:^{size:d}s}}".format(size=raw_size)
            logging.info(formatter.format("Epoch", "TrnLoss", "VldLoss"))
            for epoch in range(1, epochs + 1):
                train_losses = []
                for feed_dict in q_function_net.get_train_set():
                    _, (loss, *_) = session.run(fetches=fetches, feed_dict=feed_dict)
                    train_losses.append(loss)
                validation_losses = []
                for feed_dict in q_function_net.get_validation_set():
                    loss, *_ = session.run(fetches=q_function_net.losses, feed_dict=feed_dict)
                    validation_losses.append(loss)
                train_loss = np.mean(train_losses)
                validation_loss = np.mean(validation_losses)
                formatter = "{{:^{size:d}s}}{{:^{size:d}.4f}}{{:^{size:d}.4f}}".format(size=raw_size)
                logging.info(formatter.format("{:4d}/{:<4d}".format(epoch, epochs), train_loss, validation_loss))
                figure.plot(epoch, train_loss, ".b")
                figure.plot(epoch, validation_loss, ".r")
                if epoch % 50:
                    q_function_net.save(session)
        except SIGINTException:
            logging.error("SIGINT")
        finally:
            q_function_net.save(session)


@trace
def test(net: Net):
    pass


@trace
def run(foo: str):
    q_function_net = build()
    if foo == "train":
        train(q_function_net)
    elif foo == "restore":
        train(q_function_net, True)
    elif foo == "test":
        test(q_function_net)
