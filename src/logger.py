import logging
import re
import sys
import time


class FileHandler(logging.FileHandler):
    pattern = re.compile("(\33\[\d+m)")

    def __init__(self, filename, mode='a', encoding=None, delay=False):
        super().__init__(filename, mode, encoding, delay)

    def _open(self):
        stream = logging.FileHandler._open(self)
        write = stream.write
        stream.write = lambda buffer: write(re.sub(FileHandler.pattern, "", buffer))
        return stream


formatter = logging.Formatter("[%(asctime)s] %(message)s", datefmt='%H:%M:%S')
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
fh = FileHandler("resources/logging.log")
fh.setFormatter(formatter)
fh.setLevel(logging.DEBUG)
logger.addHandler(fh)
fh = FileHandler("resources/errors.log")
fh.setFormatter(formatter)
fh.setLevel(logging.ERROR)
logger.addHandler(fh)
sh = logging.StreamHandler(sys.stdout)
sh.setFormatter(formatter)
sh.setLevel(logging.INFO)
logger.addHandler(sh)

logger.debug("")
logger.debug(" --------------------------------------------------------")
logger.debug(time.strftime("  New session is started in %D at %T o'clock", time.gmtime()))
logger.debug(" --------------------------------------------------------")
