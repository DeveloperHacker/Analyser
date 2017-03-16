import numpy as np

from utils import dumper as _dumper
from variables import path as _path
from variables.tags import *
from variables.train import *

EMBEDDINGS = _dumper.load(_path.EMBEDDINGS)


def _embedding(word):
    return EMBEDDINGS[word] if word in EMBEDDINGS else EMBEDDINGS["UNK"]


CONSTANTS = (
    (STRING, _embedding(STRING)),
    (VARIABLE, _embedding(VARIABLE)),
    (NUMBER, _embedding(NUMBER)),
    (TRUE, _embedding(TRUE)),
    (FALSE, _embedding(FALSE)),
    (NULL, _embedding(NULL))
)

OPERATORS = (
    (EQUAL, _embedding(EQUAL)),
    (NOT_EQUAL, _embedding(NOT_EQUAL)),
    (IS, _embedding(IS)),
    (IS_NOT, _embedding(IS_NOT))
)

DELIMITERS = (
    (PUNCTUATION, _embedding(PUNCTUATION)),
    (END, _embedding(END)),
    (NOP, _embedding(NOP))
)

TOKENS = CONSTANTS + OPERATORS + DELIMITERS
NUM_TOKENS = len(TOKENS)

GO = np.zeros([EMBEDDING_SIZE], dtype=np.float32)
PAD = np.ones([EMBEDDING_SIZE], dtype=np.float32)
INITIAL_STATE = np.zeros([BATCH_SIZE, INPUT_STATE_SIZE], dtype=np.float32)
