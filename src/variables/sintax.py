from enum import Enum

import numpy as np

NUM_TOKENS = 0


class Token:
    def __init__(self, name: str):
        global NUM_TOKENS
        self.uid = NUM_TOKENS
        NUM_TOKENS += 1
        self._embedding = None
        self.name = name

    @property
    def embedding(self) -> np.ndarray:
        if self._embedding is None:
            self._embedding = np.zeros([NUM_TOKENS], dtype=np.float32)
            self._embedding[self.uid] = 1.0
        # noinspection PyTypeChecker
        return self._embedding


class Constant(Token):
    def __init__(self, name: str):
        super().__init__(name)


class Operator(Token):
    def __init__(self, name: str, args: int):
        super().__init__(name)
        self.args = args


class Delimiter(Token):
    def __init__(self, name: str):
        super().__init__(name)


class Tokens(Enum):
    STRING = Constant("string")
    VARIABLE = Constant("variable")
    NUMBER = Constant("number")
    TRUE = Constant("true")
    FALSE = Constant("false")
    NULL = Constant("null")
    EQUAL = Operator("equal", 2)
    NOT_EQUAL = Operator("not equal", 2)
    IS = Operator("is", 2)
    NOT_IS = Operator("not is", 2)
    # PUNCTUATION = Delimiter("punctuation")
    END = Delimiter("end")
    NOP = Delimiter("nop")

    @staticmethod
    def get(uid):
        return list(Tokens)[uid].value
