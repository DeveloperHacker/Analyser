import numpy as np

from variables.syntax import Tokens, Function, Constant


class Evaluator:
    @staticmethod
    def evaluate_syntax(output: list) -> np.ndarray:
        FUNCTION = 1
        ARGUMENT = 2
        END = 3
        losses = []
        for indexes in output:
            args = 0
            loss = 0.0
            state = FUNCTION
            tokens = []
            for i in range(len(indexes)):
                uid = indexes[i]
                token = Tokens.get(uid)
                tokens.append(token.name)
                if state == FUNCTION:
                    if isinstance(token, Function):
                        args = token.arguments
                        state = ARGUMENT
                    elif token == Tokens.END:
                        loss += 2.0 * (len(indexes) - i)
                        state = END
                    else:
                        loss += 10.0
                elif state == ARGUMENT:
                    if not isinstance(token, Constant):
                        loss += 10.0
                    args -= 1
                    if args == 0:
                        state = FUNCTION
                elif state == END:
                    if token != Tokens.END:
                        loss += 3.0
            if state != END:
                loss += 7.0
            losses.append(loss)
        return np.asarray(losses)

    @staticmethod
    def evaluate_correlation(inputs: list, output: list) -> np.ndarray:
        return np.zeros((len(output),), dtype=np.float32)

    @staticmethod
    def evaluate_causation(inputs: list, output: list) -> np.ndarray:
        return np.zeros((len(output),), dtype=np.float32)

    @staticmethod
    def evaluate(inputs: dict, output: list) -> np.ndarray:
        inputs = {label: np.transpose(inp, [1, 0]) for label, inp in inputs.items()}
        output = np.transpose(output, [1, 0, 2])
        output = np.argmax(output, 2)
        syntax_evaluation = Evaluator.evaluate_syntax(output)
        correlation = Evaluator.evaluate_correlation(inputs, output)
        causation = Evaluator.evaluate_causation(inputs, output)
        return syntax_evaluation + correlation + causation
