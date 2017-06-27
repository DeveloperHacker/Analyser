import logging
import os
import sys

from constants.paths import LOG


def init():
    logging.basicConfig(level=logging.INFO, filename=LOG)
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    os.environ["OUTPUT_TYPE"] = "dfs_sequence"
