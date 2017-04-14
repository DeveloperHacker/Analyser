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


Constants = []
Functions = []
Delimiters = []
Instances = []


class Constant(Token):
    def __init__(self, name: str):
        super().__init__(name)
        Constants.append(self)
        Instances.append(self)


class Function(Token):
    def __init__(self, name: str, arguments: int):
        super().__init__(name)
        self.arguments = arguments
        Functions.append(self)
        Instances.append(self)


class Delimiter(Token):
    def __init__(self, name: str):
        super().__init__(name)
        Delimiters.append(self)
        Instances.append(self)


class Tokens:
    STRING = Constant("string")
    VARIABLE = Constant("variable")
    NUMBER = Constant("number")
    TRUE = Constant("true")
    FALSE = Constant("false")
    NULL = Constant("null")
    EQUAL = Function("equal", 2)
    NOT_EQUAL = Function("not equal", 2)
    IS = Function("is", 2)
    NOT_IS = Function("not is", 2)
    END = Delimiter("end")
    # NOP = Delimiter("nop")

    @staticmethod
    def get(uid):
        return Instances[uid]
