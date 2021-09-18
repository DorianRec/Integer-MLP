import numpy as np


class Alternating:
    value = 1

    def __init__(self, value=None):
        if value is not None:
            self.value = value

    def __call__(self, shape=None):
        # TODO why could this be None?
        assert shape is not None
        if shape is None:
            raise Exception('shape must be specified! Without is not supported yet!')
        (x, y) = shape

        return np.fromfunction(lambda i, j: (-self.value) + 2 * self.value * ((i + j) % 2), (x, y))
