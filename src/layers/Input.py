import numpy as np

from layers.Layer import Layer


class Input(Layer):
    # Connected
    isOutput = False

    # Information
    shape = None

    # Only 2D shape is allowed
    def __init__(self, shape=None):
        self.name = "Input"
        if shape is None:
            raise Exception('shape must be specified! Without is not supported yet!')
        self.shape = shape

    def evaluate(self, x_test, y_test):
        # TODO check shape
        return 1

    # x_test 1D
    def predict(self, x_test):
        assert np.array_equal(self.shape, x_test.shape)
        self.o_j = x_test
        return x_test

    def get_shape(self):
        return self.shape

    def get_params(self):
        return 0
