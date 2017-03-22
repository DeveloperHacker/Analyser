import random

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
                placeholder = tf.placeholder(tf.float32, [BATCH_SIZE, EMBEDDING_SIZE], "input_%d" % i)
                inputs[label].append(placeholder)
                inputs_sizes[label] = tf.placeholder(tf.int32, [BATCH_SIZE], "input_sizes")
    with vs.variable_scope("output"):
        output = []
        for i in range(OUTPUT_SIZE):
            placeholder = tf.placeholder(tf.float32, [BATCH_SIZE, NUM_TOKENS], "output_%d" % i)
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


def most_different(samples: list, most: int):
    num_columns = len(samples) / 5
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
        if num_samples > most:
            line = i
            break
    assert line is not None
    samples = []
    for _, basket in baskets.items():
        random.shuffle(basket)
        samples.extend(basket[:min(len(basket), line)])
    return samples[:most]


def noise(array: np.ndarray, std: float):
    return array + np.random.uniform(-std, std, array.shape)


def build_samples(doc, evaluate):
    ARGUMENT = 1
    FUNCTION = 2

    TRUE_SAMPLES_NOISE_STD = 0.1
    SAMPLES_NOISE_STD = 0.5

    NUM_SAMPLES = 100
    NUM_TRUE_SAMPLES = 20
    NUM_GENETIC_CYCLES = 10
    NOISE_DEPTH = 3

    inputs = {}
    for label, (embeddings, _) in doc:
        inputs[label] = []
        for _ in range(INPUT_SIZE):
            inputs[label].append([])
        line = embeddings + [PAD for _ in range(INPUT_SIZE - len(embeddings))]
        for i, embedding in enumerate(line):
            inputs[label][i].append(embedding)
        inputs[label] = np.expand_dims(inputs[label], axis=1)
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
                    sample.append(noise(Tokens.END.embedding, TRUE_SAMPLES_NOISE_STD))
                    sample.extend([noise(Tokens.NOP.embedding, TRUE_SAMPLES_NOISE_STD)] * (arguments + expected))
                    break
                sample.append(noise(function.embedding, TRUE_SAMPLES_NOISE_STD))
                num_functions -= 1
                state = ARGUMENT
            elif state == ARGUMENT:
                i = random.randrange(len(Constants))
                constant = Constants[i]
                sample.append(noise(constant.embedding, TRUE_SAMPLES_NOISE_STD))
                arguments -= 1
                if arguments == 0:
                    state = FUNCTION
        samples.append((sample, evaluate(inputs, np.expand_dims(sample, axis=1))[0]))

    true_samples = most_different(samples, NUM_TRUE_SAMPLES)
    random.shuffle(samples)
    samples = samples[:NUM_SAMPLES - NUM_TRUE_SAMPLES]
    for i in range(NUM_GENETIC_CYCLES):
        for j in range(NUM_SAMPLES - NUM_TRUE_SAMPLES):
            sample = samples[j][0][::]
            for k in range(NOISE_DEPTH):
                n = random.randrange(NUM_TOKENS)
                m = random.randrange(len(sample))
                sample[m] = noise(Tokens.get(n).embedding, SAMPLES_NOISE_STD)
            samples.append((sample, evaluate(inputs, np.expand_dims(sample, axis=1))[0]))
        samples = most_different(samples, NUM_SAMPLES - NUM_TRUE_SAMPLES)
    return true_samples + samples


@trace
def build_feed_dicts(inputs, inputs_sizes, output, evaluation, evaluate):
    methods = dumper.load(VEC_METHODS)
    baskets = {}
    for basket, docs in batcher.throwing(methods, [INPUT_SIZE]).items():
        baskets[basket] = []
        for doc in docs:
            samples = build_samples(doc, evaluate)
            baskets[basket].extend(zip((doc for _ in range(len(samples))), samples))
        random.shuffle(baskets[basket])
    batches = {basket: batcher.chunks(data, BATCH_SIZE) for basket, data in baskets.items()}
    feed_dicts = []
    for batch in batches[INPUT_SIZE]:
        feed_dict = {}
        _output = []
        for line in output:
            feed_dict[line] = []
            _output.append([])
        for label, inp in inputs.items():
            for line in inp:
                feed_dict[line] = []
            feed_dict[inputs_sizes[label]] = []
        feed_dict[evaluation] = []
        for doc, (sample, eval) in batch:
            feed_dict[evaluation].append(eval)
            for label, (embeddings, _) in doc:
                line = embeddings + [PAD for _ in range(INPUT_SIZE - len(embeddings))]
                feed_dict[inputs_sizes[label]].append(len(embeddings))
                for i, embedding in enumerate(line):
                    feed_dict[inputs[label][i]].append(embedding)
            for i, embedding in enumerate(sample):
                feed_dict[output[i]].append(embedding)
                _output[i].append(embedding)
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

    train_set, validation_set = build_feed_dicts(inputs, inputs_sizes, output, evaluation, contract.evaluate)
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
        writer = tf.summary.FileWriter(SEQ2SEQ, session.graph)
        writer.close()
        try:
            session.run(tf.global_variables_initializer())
            if restore:
                q_function_net.restore(session)
            for epoch in range(1, epochs + 1):
                train_loss = 0.0
                for feed_dict in q_function_net.train_set:
                    _, (local_loss, *_) = session.run(fetches=fetches, feed_dict=feed_dict)
                    train_loss += local_loss
                train_loss /= len(q_function_net.train_set)
                validation_loss = 0.0
                for feed_dict in q_function_net.validation_set:
                    local_loss, *_ = session.run(fetches=q_function_net.losses, feed_dict=feed_dict)
                    validation_loss += local_loss
                validation_loss /= len(q_function_net.validation_set)
                logging.info("Epoch: {:4d}/{:<4d} TrainLoss: {:.4f} ValidationLoss: {:.4f}"
                             .format(epoch, epochs, train_loss, validation_loss))
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
