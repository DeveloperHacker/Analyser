from seq2seq.analyser import AnalyserNet
from seq2seq.q_function import QFunctionNet
from utils.wrapper import trace
from variables.sintax import *
from variables.train import CONTRACT_EPOCHS


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
    q_function_net.build_inputs()
    q_function_net.substitute_embeddings()
    analyser_net.indexes = q_function_net.indexes
    analyser_net.inputs_sizes = q_function_net.inputs_sizes
    analyser_net.substitute_embeddings()
    analyser_net.build_outputs()
    q_function_net.output = analyser_net.output
    q_function_net.build_outputs()
    analyser_net.q = q_function_net.q
    analyser_net.build_fetches()
    analyser_net.build_feed_dicts(AnalyserNet.data_set())
    q_function_net.build_fetches()
    q_function_net.build_feed_dicts(QFunctionNet.data_set())
    return analyser_net, q_function_net


@trace
def train(analyser_net: AnalyserNet, q_function_net: QFunctionNet, restore: bool = False, epochs=CONTRACT_EPOCHS):
    if not restore:
        analyser_net.pretrain()
    main_train_set = q_function_net.train_set
    main_validation_set = q_function_net.validation_set
    last_train_sets = [[]] * 3
    last_validation_sets = [[]] * 3
    for epoch in range(epochs):
        q_function_net.train(epochs=10)
        batches = analyser_net.train(q_function_net, True, epochs=10)
        last_train_sets.pop(0)
        last_validation_sets.pop(0)
        q_function_net.build_feed_dicts(batches)
        last_train_sets.append(q_function_net.train_set)
        last_validation_sets.append(q_function_net.validation_set)
        q_function_net.train_set = main_train_set[::]
        q_function_net.validation_set = main_validation_set[::]
        for train_set in last_train_sets:
            q_function_net.train_set.extend(train_set)
        for validation_set in last_validation_sets:
            q_function_net.validation_set.extend(validation_set)


@trace
def test(analyser_net, q_function_net):
    analyser_net.test()
    q_function_net.test()


@trace
def run(foo: str):
    nets = build()
    if foo == "train":
        train(*nets)
    elif foo == "restore":
        train(*nets, restore=True)
    elif foo == "test":
        test(*nets)
