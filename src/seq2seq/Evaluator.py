import numpy as np
from typing import Iterable, Dict

from variables.syntax import Tokens, Function, Constant
from variables.train import BATCH_SIZE


class Evaluator:
    @staticmethod
    def evaluate_syntax(output_indexes: Iterable) -> float:
        FUNCTION = 1
        ARGUMENT = 2
        END = 3

        args = 0
        loss = 0
        state = FUNCTION
        tokens = []
        for i, uid in enumerate(output_indexes):
            token = Tokens.get(uid)
            tokens.append(token.name)
            if state == FUNCTION:
                if isinstance(token, Function):
                    args = token.arguments
                    state = ARGUMENT
                elif token == Tokens.END:
                    loss += 0.5 * (len(output_indexes) - i)
                    state = END
                else:
                    loss += 10
            elif state == ARGUMENT:
                if not isinstance(token, Constant):
                    loss += 10
                args -= 1
                if args == 0:
                    state = FUNCTION
            elif state == END:
                if token != Tokens.END:
                    loss += 3
        if state != END:
            loss += 7
        return loss

    @staticmethod
    def evaluate_correlation(inputs_indexes: Dict[str, Iterable[int]], output_indexes: Iterable[int]) -> float:
        pass
        return 0

    @staticmethod
    def evaluate_causation(inputs_indexes: Dict[str, Iterable[int]], output_indexes: Iterable[int]) -> float:
        pass
        return 0

    @staticmethod
    def evaluate_batch(inputs: dict, output: list) -> np.ndarray:
        inputs = {label: np.transpose(inp, [1, 0]) for label, inp in inputs.items()}
        array = tuple({} for _ in range(BATCH_SIZE))
        for label, arr in inputs.items():
            for i, elem in enumerate(arr):
                array[i][label] = elem
        output = np.transpose(output, [1, 0, 2])
        losses = []
        for inputs_indexes, output_indexes in zip(array, output):
            losses.append(Evaluator.evaluate(inputs_indexes, output_indexes))
        return np.asarray(losses, np.float32)

    @staticmethod
    def evaluate(inputs_indexes: dict, output_indexes: list):
        output_indexes = np.argmax(output_indexes, 1)
        syntax_evaluation = Evaluator.evaluate_syntax(output_indexes)
        correlation = Evaluator.evaluate_correlation(inputs_indexes, output_indexes)
        causation = Evaluator.evaluate_causation(inputs_indexes, output_indexes)
        return syntax_evaluation + causation + correlation
