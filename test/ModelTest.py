import math

import numpy as np
from Model import Model
from initializers.Alternating import Alternating
from layers.Dense import Dense
from layers.Flatten import Flatten
from layers.Input import Input
import unittest


class TestStringMethods(unittest.TestCase):

    def test_upper(self):
        self.assertEqual('foo'.upper(), 'FOO')

    def test_isupper(self):
        self.assertTrue('FOO'.isupper())
        self.assertFalse('Foo'.isupper())

    def test_split(self):
        s = 'hello world'
        self.assertEqual(s.split(), ['hello', 'world'])
        # check that s.split fails when the separator is not a string
        with self.assertRaises(TypeError):
            s.split(2)

    # Test, whether 1D shape is allowed
    def test_1D_model1(self):
        inputs = Input(shape=np.array([42]))
        model = Model(inputs, inputs)
        model.compile()
        self.assertTrue(np.array_equal(model.predict(np.ones(42)), np.ones(42)))

    def test_easy_model1(self):
        inputs = Input(shape=(5, 5))
        model = Model(inputs, inputs)
        model.compile()
        self.assertTrue(np.array_equal(model.predict(np.ones((5, 5))), np.ones((5, 5))))

    def test_easy_model2(self):
        inputs = Input(shape=(5, 5))
        x = Flatten()(inputs)
        model = Model(inputs, x)
        model.compile()
        self.assertTrue(np.array_equal(model.predict(np.ones((5, 5))), np.ones(25)))

    def test_predict1(self):
        inputs = Input(shape=(2, 2))
        flatten = Flatten()(inputs)
        init = Alternating(value=1)
        relu = Dense(5, activation="relu", kernel_initializer=init)(flatten)
        model = Model(inputs, relu)
        model.compile()

        prediction = model.predict(np.ones((2, 2)))
        print(flatten.o_j)
        self.assertTrue(np.array_equal(relu.net_j, np.array([-1, 1, -1, 1, -1])))
        self.assertTrue(np.array_equal(relu.o_j, np.array([0, 1, 0, 1, 0])))

    def test_predict2(self):
        # TODO make Input(shape=n) possible
        inputs = Input(shape=(2, 1))
        x = Flatten()(inputs)
        softmax = Dense(2, activation="softmax")(x)
        model = Model(inputs, softmax)
        model.compile()

        # here we cheat!
        # Think about the bias!
        softmax.w_ij = np.array([[1, 0], [0, 0], [0,0]])
        softmax.w_ijT = softmax.w_ij.T
        prediction = model.predict(np.array([[1], [0]]))
        self.assertTrue(np.array_equal(softmax.o_j, np.array([math.e/(1+math.e), 1/(1+math.e)])))

    # def model_test1(self):
    #    inputs = Input(shape=(28, 28))
    #    x = Flatten()(inputs)
    #    x = Dense(128, activation="relu")(x)
    #    outputs = Dense(10, activation="softmax")(x)
    #    model = Model(inputs, outputs)


#
#        model.summary()
#
# opt = Standard(learning_rate=0.01)
# model.compile(loss="mse", optimizer=opt)

# model.fit(x_train, y_train)
# print(model.predict(np.ones((28, 28))))


if __name__ == '__main__':
    unittest.main()
