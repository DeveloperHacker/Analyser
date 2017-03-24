from seq2seq import analyser
from seq2seq import q_function
from seq2seq.Net import *
from utils.wrapper import trace
from variables.sintax import *
from variables.train import CONTRACT_EPOCHS, Q_FUNCTION_EPOCHS, ANALYSER_EPOCHS


def evaluate(inputs: dict, output: list):
    FUNCTION = 1
    ARGUMENT = 2
    NOP = 3
    losses = []
    output = np.transpose(output, [1, 0, 2])
    for line in output:
        args = 0
        loss = 0.0
        uids = np.argmax(line, 1)
        state = FUNCTION
        tokens = []
        for i in range(len(uids)):
            uid = uids[i]
            token = Tokens.get(uid)
            tokens.append(token.name)
            if state == FUNCTION:
                if isinstance(token, Function):
                    args = token.arguments
                    state = ARGUMENT
                elif token == Tokens.END:
                    loss += 2.0 * (len(uids) - i)
                    state = NOP
                else:
                    loss += 10.0
            elif state == ARGUMENT:
                if not isinstance(token, Constant):
                    loss += 10.0
                args -= 1
                if args == 0:
                    state = FUNCTION
            elif state == NOP:
                if token != Tokens.NOP:
                    loss += 3.0
        if state != NOP:
            loss += 7.0
        losses.append(loss)
    return losses


@trace
def build():
    analyser_net = AnalyserNet()
    q_function_net = QFunctionNet()

    inputs, inputs_sizes, initial_decoder_state, evaluation = q_function.build_inputs()
    analyser_net.inputs = inputs
    analyser_net.inputs_sizes = inputs_sizes
    analyser_net.initial_decoder_state = initial_decoder_state
    q_function_net.inputs = inputs
    q_function_net.inputs_sizes = inputs_sizes
    q_function_net.initial_decoder_state = initial_decoder_state
    q_function_net.evaluation = evaluation

    output, logits, analyser_scope = analyser.build_outputs(inputs, inputs_sizes, initial_decoder_state)
    analyser_net.output = output
    analyser_net.logits = logits
    analyser_net.scope = analyser_scope
    q_function_net.output = output

    q, q_function_scope = q_function.build_outputs(inputs, inputs_sizes, initial_decoder_state, output)
    q_function_net.q = q
    q_function_net.scope = q_function_scope

    train_set, validation_set = analyser.build_feed_dicts(inputs, inputs_sizes, initial_decoder_state)
    analyser_net.train_set = train_set
    analyser_net.validation_set = validation_set

    train_set, validation_set = q_function.build_feed_dicts(inputs, inputs_sizes, output, evaluation, evaluate)
    q_function_net.train_set = train_set
    q_function_net.validation_set = validation_set

    optimise, losses = q_function.build_fetches(q_function_scope, evaluation, q)
    q_function_net.optimise = optimise
    q_function_net.losses = losses

    optimise, losses = analyser.build_fetches(analyser_scope, q)
    analyser_net.optimise = optimise
    analyser_net.losses = losses
    return analyser_net, q_function_net


@trace
def train(analyser_net: AnalyserNet, q_function_net: QFunctionNet, restore: bool = False, epochs=CONTRACT_EPOCHS):
    # with tf.Session() as session:
    #     writer = tf.summary.FileWriter(SEQ2SEQ, session.graph)
    #     writer.close()
    #     exit(1)
    if not restore:
        analyser.pretrain(analyser_net)
    for epoch in range(epochs):
        q_function_epochs = Q_FUNCTION_EPOCHS * epoch // epochs + 10
        q_function.train(q_function_net, restore | (epoch != 0), q_function_epochs)
        analyser_epochs = ANALYSER_EPOCHS * epoch // epochs + 10
        analyser.train(analyser_net, q_function_net, True, analyser_epochs)


@trace
def test(analyser_net: AnalyserNet):
    analyser.test(analyser_net)


@trace
def run(foo: str):
    analyser_net, q_function_net = build()
    if foo == "train":
        train(analyser_net, q_function_net)
    elif foo == "restore":
        train(analyser_net, q_function_net, True)
    elif foo == "test":
        test(analyser_net, q_function_net)
