import logging
import re
import sys
import time
import types

from configurations.paths import RESOURCES

FILE_LOGGER = "file"
TIMING_LOGGER = "timing"
TABLE_LOGGER = "table"
INFO_LOGGER = "info"
GENERATOR_LOGGER = "generator"
LOG_PATH = RESOURCES + "/log.txt"


class FileHandler(logging.FileHandler):
    pattern = re.compile("(\33\[\d+m)")

    def __init__(self, filename, mode='a', encoding=None, delay=False):
        super().__init__(filename, mode, encoding, delay)

    def _open(self):
        stream = logging.FileHandler._open(self)
        write = stream.write
        stream.write = lambda buffer: write(re.sub(FileHandler.pattern, "", buffer))
        return stream


file_logger = logging.getLogger(FILE_LOGGER)
file_logger.setLevel(logging.INFO)
fh = FileHandler(LOG_PATH)
fh.setLevel(logging.INFO)
fh.setFormatter(logging.Formatter("%(message)s"))
file_logger.addHandler(fh)

timing_logger = logging.getLogger(TIMING_LOGGER)
timing_logger.setLevel(logging.INFO)
fh = FileHandler(LOG_PATH)
fh.setLevel(logging.INFO)
fh.setFormatter(logging.Formatter("[%(asctime)s] %(message)s", datefmt='%m.%d.%Y %H:%M:%S'))
sh = logging.StreamHandler(sys.stdout)
sh.setFormatter(logging.Formatter("[%(asctime)s] %(message)s", datefmt='%m.%d.%Y %H:%M:%S'))
sh.setLevel(logging.INFO)
timing_logger.addHandler(sh)
timing_logger.addHandler(fh)

table_logger = logging.getLogger(TABLE_LOGGER)
table_logger.setLevel(logging.INFO)
fh = FileHandler(LOG_PATH)
fh.setFormatter(logging.Formatter("%(message)s"))
fh.setLevel(logging.INFO)
sh = logging.StreamHandler(sys.stdout)
sh.setFormatter(logging.Formatter("%(message)s"))
sh.setLevel(logging.INFO)
table_logger.addHandler(sh)
table_logger.addHandler(fh)

info_logger = logging.getLogger(INFO_LOGGER)
info_logger.setLevel(logging.INFO)
fh = FileHandler(LOG_PATH)
fh.setFormatter(logging.Formatter("[%(asctime)s] %(message)s", datefmt='%m.%d.%Y %H:%M:%S'))
fh.setLevel(logging.INFO)
sh = logging.StreamHandler(sys.stderr)
sh.setFormatter(logging.Formatter("[%(asctime)s] %(message)s", datefmt='%m.%d.%Y %H:%M:%S'))
sh.setLevel(logging.INFO)
info_logger.addHandler(sh)
info_logger.addHandler(fh)

__init__ = types.SimpleNamespace()
if not hasattr(__init__, "inited"):
    setattr(__init__, "inited", True)
    file_logger.info("")
    file_logger.info(" --------------------------------------------------------")
    file_logger.info(time.strftime("  New session is started in %D at %T o'clock", time.gmtime()))
    file_logger.info(" --------------------------------------------------------")
