class SmoothedValue:
    @property
    def value(self):
        return self._value

    @value.setter
    def value(self, value):
        if self._value is None:
            self._value = value
        else:
            self._value = self._smoothing * self._value + (1 - self._smoothing) * value

    def __init__(self, smoothing: float, initial_value: float = None):
        self._value = initial_value
        self._smoothing = smoothing

    def __call__(self, value: float = None):
        if value is not None:
            self.value = value
        return self.value
