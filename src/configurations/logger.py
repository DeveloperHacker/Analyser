import logging
import sys

from configurations.paths import RESOURCES

TIMING_LOGGER = "timing"
TABLE_LOGGER = "table"
INFO_LOGGER = "info"
GENERATOR_LOGGER = "generator"
LOG_PATH = RESOURCES + "/log.txt"

timing_logger = logging.getLogger(TIMING_LOGGER)
timing_logger.setLevel(logging.INFO)
fh = logging.FileHandler(LOG_PATH)
fh.setLevel(logging.INFO)
fh.setFormatter(logging.Formatter("[%(asctime)s] %(message)s", datefmt='%m.%d.%Y %H:%M:%S'))
sh = logging.StreamHandler(sys.stdout)
sh.setFormatter(logging.Formatter("[%(asctime)s] %(message)s", datefmt='%m.%d.%Y %H:%M:%S'))
sh.setLevel(logging.INFO)
timing_logger.addHandler(sh)
timing_logger.addHandler(fh)

table_logger = logging.getLogger(TABLE_LOGGER)
table_logger.setLevel(logging.INFO)
fh = logging.FileHandler(LOG_PATH)
fh.setFormatter(logging.Formatter("%(message)s"))
fh.setLevel(logging.INFO)
sh = logging.StreamHandler(sys.stdout)
sh.setFormatter(logging.Formatter("%(message)s"))
sh.setLevel(logging.INFO)
table_logger.addHandler(sh)
table_logger.addHandler(fh)

info_logger = logging.getLogger(INFO_LOGGER)
info_logger.setLevel(logging.INFO)
fh = logging.FileHandler(LOG_PATH)
fh.setFormatter(logging.Formatter("[%(asctime)s] %(message)s", datefmt='%m.%d.%Y %H:%M:%S'))
fh.setLevel(logging.INFO)
sh = logging.StreamHandler(sys.stderr)
sh.setFormatter(logging.Formatter("[%(asctime)s] %(message)s", datefmt='%m.%d.%Y %H:%M:%S'))
sh.setLevel(logging.INFO)
info_logger.addHandler(sh)
info_logger.addHandler(fh)
