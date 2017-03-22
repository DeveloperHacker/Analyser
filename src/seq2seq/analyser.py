from seq2seq import contract
from seq2seq import q_function
from seq2seq.Net import *
from seq2seq.seq2seq import *
from utils import batcher, dumper
from utils.Figure import Figure
from utils.handlers import SIGINTException
from utils.wrapper import *
from variables.embeddings import *
from variables.path import *
from variables.sintax import NUM_TOKENS
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
                placeholder = tf.placeholder(tf.float32, [BATCH_SIZE, EMBEDDING_SIZE], "batch_%d" % i)
                inputs[label].append(placeholder)
                inputs_sizes[label] = tf.placeholder(tf.int32, [BATCH_SIZE], "input_sizes")
    return inputs, inputs_sizes


@trace
def build_outputs(inputs, inputs_sizes):
    with vs.variable_scope("analyser"):
        inputs_states = []
        for label in PARTS:
            with vs.variable_scope(label):
                output_states_fw, output_states_bw = build_encoder(inputs[label], INPUT_STATE_SIZE, inputs_sizes[label])
                inputs_states.append(tf.concat(axis=1, values=[output_states_fw[0], output_states_bw[-1]]))
                inputs_states.append(tf.concat(axis=1, values=[output_states_fw[-1], output_states_bw[0]]))
        goes = tf.stack([GO for _ in range(BATCH_SIZE)])
        decoder_inputs = [goes for _ in range(OUTPUT_SIZE)]
        inputs_states = tf.stack(inputs_states)
        inputs_states = tf.transpose(inputs_states, [1, 0, 2])
        output, _ = build_decoder(decoder_inputs, inputs_states, INPUT_STATE_SIZE, INITIAL_STATE, EMBEDDING_SIZE,
                                  loop=True)
        with vs.variable_scope("softmax"):
            W_shape = [EMBEDDING_SIZE, NUM_TOKENS]
            B_shape = [NUM_TOKENS]
            std = INITIALIZATION_STD
            W = tf.Variable(initial_value=tf.truncated_normal(W_shape, stddev=std, dtype=tf.float32), name="weights")
            B = tf.Variable(initial_value=tf.truncated_normal(B_shape, stddev=std, dtype=tf.float32), name="biases")
            output = tf.reshape(tf.stack(output), [BATCH_SIZE * OUTPUT_SIZE, EMBEDDING_SIZE])
            logits = tf.reshape(tf.matmul(output, W) + B, [OUTPUT_SIZE, BATCH_SIZE, NUM_TOKENS])
            output = tf.nn.softmax(logits)
            output = tf.unstack(output)
        scope = vs.get_variable_scope().name
    return output, logits, scope


@trace
def build_fetches(analyser_scope, q):
    with vs.variable_scope("loss"):
        trainable_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, analyser_scope)
        # regularization_variables = [analyser_scope + "/" + variable for variable in REGULARIZATION_VARIABLES]
        # l2_loss = build_l2_loss(trainable_variables, regularization_variables)
        # q = tf.reduce_mean(q)
        # loss = Q_WEIGHT * q + L2_WEIGHT * l2_loss
        loss = tf.reduce_mean(q)
    with vs.variable_scope("optimiser"):
        optimise = tf.train.AdamOptimizer(beta1=0.90).minimize(loss, var_list=trainable_variables)
    return optimise, (loss,)


@trace
def build_feed_dicts(inputs, inputs_sizes):
    methods = dumper.load(VEC_METHODS)
    baskets = batcher.throwing(methods, [INPUT_SIZE])
    batches = {basket: batcher.build_batches(data, BATCH_SIZE) for basket, data in baskets.items()}
    feed_dicts = []
    for batch in batches[INPUT_SIZE]:
        feed_dict = {}
        for label, (lines, _) in batch.items():
            for _LINE in inputs[label]:
                feed_dict[_LINE] = []
            for embeddings in lines:
                line = embeddings + [PAD for _ in range(INPUT_SIZE - len(embeddings))]
                for i, embedding in enumerate(line):
                    feed_dict[inputs[label][i]].append(embedding)
            feed_dict[inputs_sizes[label]] = tuple(len(emb) for emb in lines)
        feed_dicts.append(feed_dict)
    train_set = feed_dicts[:len(feed_dicts) * TRAIN_SET]
    validation_set = feed_dicts[len(train_set):]
    return train_set, validation_set


@trace
def build() -> (AnalyserNet, QFunctionNet):
    analyser_net = AnalyserNet()
    q_function_net = QFunctionNet()

    inputs, inputs_sizes, _, evaluation = q_function.build_inputs()
    analyser_net.inputs = inputs
    analyser_net.inputs_sizes = inputs_sizes
    q_function_net.inputs = inputs
    q_function_net.inputs_sizes = inputs_sizes
    q_function_net.evaluation = evaluation

    output, logits, analyser_scope = build_outputs(inputs, inputs_sizes)
    analyser_net.output = output
    analyser_net.logits = logits
    analyser_net.scope = analyser_scope
    q_function_net.output = output

    q, q_function_scope = q_function.build_outputs(inputs, inputs_sizes, output)
    q_function_net.q = q
    q_function_net.scope = q_function_scope

    train_set, validation_set = build_feed_dicts(inputs, inputs_sizes)
    analyser_net.train_set = train_set
    analyser_net.validation_set = validation_set

    optimise, losses = build_fetches(analyser_scope, q)
    analyser_net.optimise = optimise
    analyser_net.losses = losses
    return analyser_net, q_function_net


@trace
def pretrain(analyser_net: AnalyserNet):
    with tf.Session() as session:
        session.run(tf.global_variables_initializer())
        analyser_net.save(session)


@trace
def train(analyser_net: AnalyserNet, q_function_net: QFunctionNet, restore: bool = False, epochs=ANALYSER_EPOCHS):
    fetches = (
        analyser_net.optimise,
        analyser_net.inputs,
        analyser_net.output,
        analyser_net.losses
    )
    with tf.Session() as session, tf.device('/cpu:0'), Figure(xauto=True) as figure:
        writer = tf.summary.FileWriter(SEQ2SEQ, session.graph)
        writer.close()
        try:
            session.run(tf.global_variables_initializer())
            q_function_net.restore(session)
            if restore:
                analyser_net.restore(session)
            for epoch in range(epochs):
                losses = (0.0 for _ in range(len(analyser_net.losses)))
                evaluation = 0.0
                for feed_dict in analyser_net.train_set:
                    _, inputs, output, local_losses = session.run(fetches=fetches, feed_dict=feed_dict)
                    evaluation += np.mean(contract.evaluate(inputs, output))
                    losses = [local_losses[i] + loss for i, loss in enumerate(losses)]
                losses = [loss / len(analyser_net.train_set) for loss in losses]
                loss = losses[0]
                evaluation /= len(analyser_net.train_set)
                logging.info("Epoch: {:4d}/{:<4d} Loss: {:.4f} Eval: {:.4f}".format(epoch, epochs, loss, evaluation))
                figure.plot(epoch, loss)
                if epoch % 50:
                    analyser_net.save(session)
        except SIGINTException:
            logging.error("SIGINT")
        finally:
            analyser_net.save(session)


@trace
def test(analyser_net: AnalyserNet):
    embeddings = list(EMBEDDINGS)
    first = [analyser_net.inputs[label][0] for label in PARTS]
    with tf.Session() as session:
        analyser_net.restore(session)
        errors = []
        roundLoss = []
        loss = []
        res_loss = []
        for _feed_dict in analyser_net.train_set:
            result = session.run(fetches=[analyser_net.output, loss] + first, feed_dict=_feed_dict)
            outputs = result[0]
            res_loss.append(result[1])
            targets = result[2:]
            assert len(targets) == 4
            for output, target in zip(outputs[:4], targets):
                for out, tar in zip(output, target):
                    i = np.argmin([np.linalg.norm(out - embedding) for embedding in embeddings])
                    errors.append(np.linalg.norm(embeddings[i] - tar) > 1e-6)
                    roundLoss.append(np.linalg.norm(embeddings[i] - tar))
                    loss.append(np.linalg.norm(out - tar))
        logging.info("Accuracy: {}%".format((1 - np.mean(errors)) * 100))
        logging.info("RoundLoss: {}".format(np.mean(roundLoss)))
        logging.info("Loss: {}".format(np.mean(loss)))
        logging.info("ResLoss: {}".format(np.mean(res_loss)))


@trace
def run(foo: str):
    analyser_net, q_function_net = build()
    if foo == "train":
        train(analyser_net, q_function_net)
    elif foo == "restore":
        train(analyser_net, q_function_net, True)
    elif foo == "test":
        test(analyser_net)
