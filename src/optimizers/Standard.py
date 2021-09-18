import numpy as np

from layers.Dense import Dense
from layers.Flatten import Flatten
from layers.Input import Input


class Standard:
    # eta
    learning_rate = 0.01

    def __init__(self, learning_rate=None):
        if learning_rate is not None:
            self.learning_rate = learning_rate

    # x_train and y_train is one dataset
    def backpropagation(self, layer, x_train, y_train, loss):
        if isinstance(layer, Input):
            pass
        elif isinstance(layer, Flatten):
            pass
        elif isinstance(layer, Dense):
            # partial E/partial w_ij = partial E/partial o_j * partial o_j/partial net_j * partial net_j/partial w_ij

            # NOW: partial E/partial o_j : Depends on E
            a = np.empty(layer.output_shape)
            if not layer.isOutput:
                for dJ in range(layer.output_shape):
                    a[dJ] = np.dot(layer.nxt.w_ij[dJ], layer.nxt.delta_j)
            elif loss == "mse":
                # TODO or is this 1/2 ?
                # E(y,o) = 1/n sum_k (y_k - o_k)^2
                # partial E/partial o_j = 2/n * (o_j - y_j)
                a = 2 / layer.output_shape * (layer.o_j - y_train)
            else:
                # simulate "mse"
                a = 2 / layer.output_shape * (layer.o_j - y_train)

            # NOW: partial o_j/partial net_j : depends on activation function
            b = layer.activation.derivative(layer.net_j, layer.o_j)

            layer.delta_j = a * b

            # NOW: partial net_j/partial w_ij = o_i
            # DeltaW_ij = -eta * partial E/partial w_ij = -eta * o_i * delta_j
            # TODO also try component-wise
            DeltaW_ij = np.empty(layer.w_ij.shape)
            for dI in range(layer.w_ij.shape[0]):
                if (dI < layer.w_ij.shape[0] - 1):
                    DeltaW_ij[dI] = (-self.learning_rate * layer.delta_j * layer.prev.o_j[dI]).ravel()
                else:  # dI=layer.w_ij.shape[0]-1, this is the bias
                    DeltaW_ij[dI] = (-self.learning_rate * layer.delta_j).ravel()
            layer.w_ij += DeltaW_ij
        else:
            raise Exception('Layer subclass not supported!')
