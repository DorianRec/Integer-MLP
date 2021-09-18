import math

import numpy as np

from layers.Layer import Layer


class Flatten(Layer):
    # Content
    # TODO is neurons needed ?
    neurons = 0
    w_ij = None

    def __init__(self):
        self.name = "Flatten"
        pass

    def __call__(self, prev):
        # Connect
        self.prev = prev
        prev.nxt = self

        return self

    def get_shape(self):
        return math.prod(self.prev.shape)

    def get_params(self):
        return 0

    def predict(self, x_test):
        self.o_j = self.prev.predict(x_test).flatten()
        return self.o_j
