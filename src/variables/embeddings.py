from utils import dumper as _dumper
from variables import path as _path
from variables.sintax import *

EMBEDDINGS = _dumper.load(_path.EMBEDDINGS)


def embedding(word):
    return EMBEDDINGS[word] if word in EMBEDDINGS else EMBEDDINGS["UNK"]


CONSTANTS = (
    (STRING, embedding(STRING)),
    (VARIABLE, embedding(VARIABLE)),
    (NUMBER, embedding(NUMBER)),
    (TRUE, embedding(TRUE)),
    (FALSE, embedding(FALSE)),
    (NULL, embedding(NULL))
)

OPERATORS = (
    (EQUAL, embedding(EQUAL)),
    (NOT_EQUAL, embedding(NOT_EQUAL)),
    (IS, embedding(IS)),
    (IS_NOT, embedding(IS_NOT))
)

DELIMITERS = (
    (PUNCTUATION, embedding(PUNCTUATION)),
    (END, embedding(END)),
    (NOP, embedding(NOP))
)

TOKENS = CONSTANTS + OPERATORS + DELIMITERS
