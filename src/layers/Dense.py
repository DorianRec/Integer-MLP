import math

import numpy as np

from activation_functions.ReLu import ReLu
from activation_functions.Softmax import Softmax
from layers.Layer import Layer
from initializers.Alternating import Alternating


class Dense(Layer):
    # Content
    activation = None
    w_ij = None
    w_ijT = None
    net_j = 0

    def __init__(self, output_shape, activation=None, kernel_initializer=None):
        self.output_shape = output_shape

        # Choose activation function
        if activation is None:
            pass
        if activation == "relu":
            self.name = "ReLu"
            self.activation = ReLu()
            # TODO implement
            pass
        if activation == "softmax":
            self.name = "Softmax"
            self.activation = Softmax()
            # TODO implement
            pass

        # Handle kernel_initializer
        if kernel_initializer is None:
            self.kernel_initializer = Alternating()
        else:
            self.kernel_initializer = kernel_initializer

    def __call__(self, prev):
        # Connect
        self.prev = prev
        prev.nxt = self

        # Initialize
        self.w_ij = self.kernel_initializer((self.prev.get_shape() + 1, self.output_shape))
        self.w_ijT = self.w_ij.T

        return self

    # Computes delta_j DeltaW_ij and updates w_ij and w_ijT
    def fit(self, x_train, y_train):
        # train previous one
        self.prev.fit(x_train, y_train)



    def predict(self, x_test):
        # Dont forget to append the bias=1
        self.net_j = np.matmul(self.w_ijT, np.append(self.prev.predict(x_test), 1)[np.newaxis].T).squeeze()
        self.o_j = self.activation.activate(self.net_j)
        return self.o_j

    def get_shape(self):
        return self.w_ij.shape[1]

    def get_params(self):
        return math.prod(self.w_ij.shape)

    # Returns the weights
    def get_weights(self):
        return self.w_ij