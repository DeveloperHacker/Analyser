from abc import abstractmethod
from multiprocessing import Value

from cmdmanager.traceable import Traceable


class Task(Traceable):
    @property
    def name(self):
        return self._name

    def __init__(self, name: str):
        super().__init__()
        self._name = name or "task"

    @abstractmethod
    def is_done(self):
        pass

    @abstractmethod
    def completeness(self) -> float:
        pass

    def value(self):
        return self.completeness()


class SimpleTask(Task):
    def __init__(self, name: str):
        super(SimpleTask, self).__init__(name)
        self._is_done = Value("i", False)

    def is_done(self):
        return self._is_done.value

    def _update(self):
        self._is_done.value = True

    def _reset(self):
        self._is_done.value = False

    def completeness(self) -> float:
        return float(self._is_done.value)


class CounterTask(Task):
    def __init__(self, progress_length: int, name: str):
        super(CounterTask, self).__init__(name)
        self._progress = Value("i", 0)
        self._length = progress_length

    def is_done(self):
        return self._progress.value >= self._length

    def _update(self):
        with self._progress.get_lock():
            self._progress.value += 1

    def _reset(self):
        with self._progress.get_lock():
            self._progress.value = 0

    def completeness(self) -> float:
        return self._progress.value / self._length


class MultiTask(Task):
    def __init__(self, name: str, *tasks):
        super(MultiTask, self).__init__(name)
        self._sub_tasks = []
        for task in tasks:
            task.add(self.update)
            self._sub_tasks.append(task)

    def is_done(self):
        return all((task.is_done() for task in self._sub_tasks))

    def _update(self):
        pass

    def _reset(self):
        for task in self._sub_tasks:
            task.reset()

    def completeness(self) -> float:
        if len(self._sub_tasks) == 0: return 1.0
        acc = 0.0
        for task in self._sub_tasks:
            acc += task.completeness()
        return acc / len(self._sub_tasks)
