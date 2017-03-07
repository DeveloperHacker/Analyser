import logging
import os
import sys
import time
from multiprocessing import Value

from tasks import Task, CounterTask
from utils import expand, unique_integer_key

from cmdmanager.items import Item, ProgressBar, Timer


class _Commands:
    info = "info"
    print = "print"
    show = "show"
    repaint_item = "repaint"


class Logger:
    def __init__(self, name=None):
        self._writer = Writer(name)
        self.add_stream(sys.stdout)

    def add_stream(self, stream):
        self._writer.add_stream(stream)

    def trace(self, foo):
        def wrapper(*args, **kwargs):
            name = foo.__name__
            self.info("Function \"{}\" is invoked".format(name))
            clock = time.time()
            result = foo(*args, **kwargs)
            delay = time.time() - clock
            self.info("Function \"{}\" is ended".format(name))
            self.info("Work time {}".format(expand(delay)))
            return result

        return wrapper

    def info(self, obj):
        self._writer.register_command(_Commands.info, obj)

    def print(self, obj):
        self._writer.register_command(_Commands.print, obj)

    def progress(self, task: Task, length: int):
        return self._writer.progress(task, length)

    def progress_iterator(self, arr: list, length: int):
        task = CounterTask(len(arr), "progress")
        progress = self._writer.progress(task, length)
        self.show(progress)
        for elem in arr:
            yield elem
            task.update()

    def timer(self, task: Task):
        return self._writer.timer(task)

    def show(self, *items):
        self._writer.register_command(_Commands.show, *items)


class Writer(logging.Handler):
    def __init__(self, name=None):
        logging.Handler.__init__(self)
        self._logger = logging.getLogger(name)
        self._logger.addHandler(self)
        self._commands = {}
        self._streams = []
        self._items = []
        self._y_cursor_position = Value("i", 2)
        self._x_cursor_position = Value("i", 1)

    def add_stream(self, stream):
        self._streams.append(stream)

    def register_command(self, command: str, *args):
        uid = unique_integer_key(self._commands)
        self._commands[uid] = (command, tuple(args))
        self._logger.log(logging.CRITICAL, uid)

    # noinspection PyBroadException
    def emit(self, record):
        try:
            uid = int(self.format(record))
            command, args = self._commands[uid]
            if command == _Commands.info:
                self._info(*args)
            elif command == _Commands.print:
                self._print(*args)
            elif command == _Commands.show:
                self._show(*args)
            elif command == _Commands.repaint_item:
                self._repaint_item(*args)
        except Exception:
            self.handleError(record)

    # Todo: normal
    @property
    def y_cursor_position(self):
        return self._y_cursor_position.value

    # Todo: normal
    @property
    def x_cursor_position(self):
        return self._x_cursor_position.value

    # Todo: remove this operation
    def _set_y_cursor_position(self, y: int):
        self._y_cursor_position.value = y

    # Todo: remove this operation
    def _set_x_cursor_position(self, x: int):
        self._x_cursor_position.value = x

    @property
    def width(self):
        term_size = os.get_terminal_size()
        return term_size.columns

    @property
    def height(self):
        term_size = os.get_terminal_size()
        return term_size.lines

    def _info(self, obj):
        self._writeln("INFO: " + str(obj))

    def _print(self, obj):
        self._writeln(">>    " + str(obj))

    def _show(self, *items):
        for item in list(items):
            if isinstance(item, str):
                self._write(item)
            elif isinstance(item, Item):
                item.set_position(self.x_cursor_position, self.y_cursor_position)
                # fixme                                                           v
                self._set_x_cursor_position(self.x_cursor_position + item.width + 2)
                self._repaint_item(item)
            else:
                raise Exception("Item or string expected")
        self._write("\n")

    def progress(self, task: Task, length: int):
        progress_bar = ProgressBar(1, -1, length, task, self._repaint_item)
        self._items.append(progress_bar)
        return progress_bar

    def timer(self, task: Task):
        timer = Timer(task, 1, -1, self._repaint_item)
        self._items.append(timer)
        return timer

    def _repaint_item(self, item: Item):
        self.acquire()
        if item.y > 0:
            go_to_item_pos = "\033[{};{}H".format(item.y, item.x)
            go_to_cursor_pos = "\033[{};{}H".format(self.y_cursor_position, self.x_cursor_position)
            line = self._truncate(item.to_line(self.width))
            self._log(go_to_item_pos + " " * item.width)
            self._log(go_to_item_pos + line + go_to_cursor_pos)
        self.release()

    def _writeln(self, text: str):
        self._write(text + "\n")

    def _write(self, text: str):
        self.acquire()
        width = self.width
        lines = []
        for line in text.split('\n'):
            while len(line) > width:
                lines.append(line[:width])
                line = line[width:]
            lines.append(line)
        prev = self.x_cursor_position
        self._set_x_cursor_position(len(lines[-1]))
        if len(lines) == 1:
            self._set_x_cursor_position(prev + self.x_cursor_position)
        self._log("\n".join(lines))
        shift = self.y_cursor_position + len(lines) - 1
        self._set_y_cursor_position(min(shift, self.height))
        shift = max(shift - self.height, 0)
        for i, item in enumerate(self._items):
            item.set_y(item.y - shift)
            self._repaint_item(item)
        self.release()

    def _truncate(self, string: str):
        return string[:min(len(string), self.width)]

    def _log(self, string: str):
        for stream in self._streams:
            stream.write(string)
            if stream and hasattr(stream, "flush"):
                stream.flush()


logger = Logger()
