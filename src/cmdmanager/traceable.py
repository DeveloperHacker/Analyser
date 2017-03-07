from abc import abstractmethod, ABCMeta

from cmdmanager.utils import unique_integer_key


class Traceable(metaclass=ABCMeta):
    @abstractmethod
    def value(self):
        pass

    def __init__(self):
        self._update_listeners = {}

    def add(self, listener) -> int:
        uid = unique_integer_key(self._update_listeners)
        self._update_listeners[uid] = listener
        return uid

    def remove(self, uid: int):
        del self._update_listeners[uid]

    def update(self):
        self._update()
        for key, listener in self._update_listeners.items():
            listener()

    def reset(self):
        self._reset()
        for key, listener in self._update_listeners.items():
            listener()

    @abstractmethod
    def _update(self):
        pass

    @abstractmethod
    def _reset(self):
        pass
