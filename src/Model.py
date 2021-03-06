from tabulate import tabulate

from optimizers.Standard import Standard
import tensorflow as tf


class Model:
    inputs = None
    output = None

    # fitting
    loss = "mse"
    optimizer = Standard(learning_rate=0.01)

    def __init__(self, inputs, output):
        self.inputs = inputs
        self.output = output
        self.output.isOutput = True

    ### Building

    def summary(self):
        assert self.inputs is not None

        data = []
        layer = self.inputs
        while layer is not None:
            data.append(layer.summary())
            layer = layer.nxt

        print(tabulate(data, headers=['Layer type', 'Output Shape', 'Param #']))

    def compile(self, loss="", optimizer=None):
        if loss == "mse":
            self.loss = loss
        else:
            pass

        if optimizer is not None:
            self.optimizer = optimizer

    ### Train the model

    def fit(self, x_train, y_train):
        assert x_train.shape[0] == y_train.shape[0]
        for i in range(x_train.shape[0]):
            # result will not be used, since results are saved in the layers
            prediction = self.predict(x_train[i])
            # TODO remove output
            print('Iteration: ' + str(i))
            print('Prediction:')
            print(prediction)
            print('Actual:')
            print(y_train[i])
            print('mse')
            print(tf.keras.losses.MeanSquaredError()(prediction, y_train[i]))
            layer = self.output
            while layer is not None:
                self.optimizer.backpropagation(layer, x_train[i], y_train[i], self.loss)
                layer = layer.prev

    ### Evaluate the model

    def evaluate(self, x_test, y_test):
        return self.output.evaluate(x_test, y_test) + 1

    # TODO implement multiple inputs.
    def predict(self, x_test):
        return self.output.predict(x_test)
