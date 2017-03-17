from seq2seq import analyser
from seq2seq import q_function
from seq2seq.Net import *
from utils.wrapper import trace
from variables.sintax import *
from variables.train import CONTRACT_EPOCHS


@trace
def build_evaluator():
    def evaluate(inputs: dict, output: list):
        losses = []
        output = np.transpose(output, [1, 0, 2])
        for line in output:
            args = 0
            loss = 0.0
            uids = np.argmax(line, 1)
            state = "operation"
            for i in range(len(uids)):
                uid = uids[i]
                token = Tokens.get(uid)
                if state == "operation":
                    if isinstance(token, Operator):
                        args = token.args
                    elif token == Tokens.END:
                        loss += 2.0 / (len(uids) - i)
                        state = "nop"
                    else:
                        loss += 10.0 / (i + 1)
                elif state == "argument":
                    if not isinstance(token, Constant):
                        loss += 10.0 / (i + 1)
                    args -= 1
                    if args == 0:
                        state = "operation"
                elif state == "nop":
                    if token != Tokens.NOP:
                        loss += 10.0
            if state != "nop":
                loss += 7.0
            losses.append(loss)
        return losses

    return evaluate


@trace
def build():
    inputs = q_function.build_placeholders()
    analyser_net = analyser.build_net(inputs)
    analyser_net.inputs.output = analyser_net.outputs.output
    q_function_net = q_function.build_net(analyser_net.inputs)
    q_function_net.fetches = q_function.build_fetches(q_function_net)
    analyser_net.fetches = analyser.build_fetches(q_function_net)
    batches = analyser.build_batches()
    analyser_net.feed_dicts = analyser.build_feed_dicts(analyser_net, batches)
    evaluate = build_evaluator()
    q_function_net.feed_dicts = q_function.build_feed_dicts(analyser_net.feed_dicts, analyser_net, evaluate)
    return analyser_net, q_function_net


@trace
def train(analyser_net: Net, q_function_net: Net, restore: bool = False):
    if not restore:
        analyser.pretrain(analyser_net)
    for epoch in range(CONTRACT_EPOCHS):
        q_function.train(q_function_net, restore | (epoch != 0))
        analyser.train(analyser_net, q_function_net, True)


@trace
def test(analyser_net: Net):
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
