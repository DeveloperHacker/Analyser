import numpy as np
from contracts.tokens import tokens
from contracts.tokens.MarkerToken import MarkerToken
from typing import List, Dict, Tuple

from utils import Dumper
from constants.generator import EMBEDDING_SIZE
from constants.paths import EMBEDDINGS

GO = "@go"
GO_emb = np.ones([EMBEDDING_SIZE], dtype=np.float32)
PAD = "@pad"
PAD_emb = np.zeros([EMBEDDING_SIZE], dtype=np.float32)
NOP = "@nop"
NOP_token = MarkerToken(NOP)
tokens.register(NOP_token)


# ToDo: thread save
class WordEmbeddings:
    _instance = None
    _word2idx = None
    _emb2idx = None

    @staticmethod
    def instance() -> dict:
        if WordEmbeddings._instance is None:
            WordEmbeddings._instance = Dumper.pkl_load(EMBEDDINGS)
            WordEmbeddings._instance[GO] = GO_emb
            WordEmbeddings._instance[PAD] = PAD_emb
            WordEmbeddings._instance = list(WordEmbeddings._instance.items())
            WordEmbeddings._instance.sort(key=lambda x: x[0])
        return WordEmbeddings._instance

    @staticmethod
    def idx2word() -> List[str]:
        return [word for word, embedding in WordEmbeddings.instance()]

    @staticmethod
    def idx2emb() -> List[np.ndarray]:
        return [embedding for word, embedding in WordEmbeddings.instance()]

    @staticmethod
    def emb2idx() -> dict:
        if WordEmbeddings._emb2idx is None:
            WordEmbeddings._emb2idx = {}
            for index, (word, embedding) in enumerate(WordEmbeddings.instance()):
                WordEmbeddings._emb2idx[tuple(embedding)] = index
        return WordEmbeddings._emb2idx

    @staticmethod
    def word2idx() -> dict:
        if WordEmbeddings._word2idx is None:
            WordEmbeddings._word2idx = {word: i for i, (word, embedding) in enumerate(WordEmbeddings.instance())}
        return WordEmbeddings._word2idx

    @staticmethod
    def get_store(key):
        if key is None:
            index = WordEmbeddings.word2idx()[PAD]
        elif isinstance(key, int):
            index = key
        elif isinstance(key, str):
            if key in WordEmbeddings.word2idx():
                index = WordEmbeddings.word2idx()[key]
            else:
                index = WordEmbeddings.word2idx()["UNK"]
        else:
            key = tuple(key)
            if key in WordEmbeddings.emb2idx():
                index = WordEmbeddings.emb2idx()[key]
            else:
                raise Exception("Store with embedding {} is not found".format(key))
        return (index,) + WordEmbeddings.instance()[index]

    @staticmethod
    def get_embedding(key):
        return WordEmbeddings.get_store(key)[2]

    @staticmethod
    def get_index(key):
        return WordEmbeddings.get_store(key)[0]

    @staticmethod
    def get_word(key):
        return WordEmbeddings.get_store(key)[1]


# ToDo: thread save
class TokenEmbeddings:
    _idx2token = None
    _token2idx = None
    _emb2idx = None
    _idx2emb = None

    @staticmethod
    def instance() -> List[str]:
        if TokenEmbeddings._idx2token is None:
            TokenEmbeddings._idx2token = list(tokens.instances())
            TokenEmbeddings._idx2token.sort()
        return TokenEmbeddings._idx2token

    @staticmethod
    def idx2token() -> List[str]:
        return TokenEmbeddings.instance()

    @staticmethod
    def idx2emb() -> List[np.ndarray]:
        if TokenEmbeddings._idx2emb is None:
            TokenEmbeddings._idx2emb = list(np.eye(len(TokenEmbeddings.instance())))
        return TokenEmbeddings._idx2emb

    @staticmethod
    def emb2idx() -> Dict[tuple, int]:
        if TokenEmbeddings._emb2idx is None:
            TokenEmbeddings._emb2idx = {tuple(embedding): i for i, embedding in enumerate(TokenEmbeddings.idx2emb())}
        return TokenEmbeddings._emb2idx

    @staticmethod
    def token2idx() -> Dict[str, int]:
        if TokenEmbeddings._token2idx is None:
            TokenEmbeddings._token2idx = {name: index for index, name in enumerate(TokenEmbeddings.instance())}
        return TokenEmbeddings._token2idx

    @staticmethod
    def get_store(key) -> Tuple[int, str, np.ndarray]:
        if key is None:
            raise ValueError
        if isinstance(key, int):
            index = key
        elif isinstance(key, str):
            if key in TokenEmbeddings.token2idx():
                index = TokenEmbeddings.token2idx()[key]
            else:
                raise Exception("Store with name {} is not found".format(key))
        else:
            key = tuple(key)
            if key in TokenEmbeddings.emb2idx():
                index = TokenEmbeddings.emb2idx()[key]
            else:
                raise Exception("Store with embedding {} is not found".format(key))
        return index, TokenEmbeddings.idx2token()[index], TokenEmbeddings.idx2emb()[index]

    @staticmethod
    def get_embedding(key) -> np.ndarray:
        return TokenEmbeddings.get_store(key)[2]

    @staticmethod
    def get_index(key) -> int:
        return TokenEmbeddings.get_store(key)[0]

    @staticmethod
    def get_token(key) -> str:
        return TokenEmbeddings.get_store(key)[1]


NUM_TOKENS = len(TokenEmbeddings.instance())
NUM_WORDS = len(WordEmbeddings.instance())
