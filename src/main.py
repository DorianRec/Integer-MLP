import numpy as np
from tensorflow import keras

from layers.Flatten import Flatten
from layers.Input import Input
from layers.Dense import Dense
from Model import Model

# Model / data parameters
from optimizers.Standard import Standard

num_classes = 10
input_shape = (28, 28, 1)

# the data, split between train and test sets
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
# TODO remove (test purposes)
(x_train, y_train), (x_test, y_test) = (x_train[0:3001], y_train[0:3001]), (x_test, y_test)
# Scale images to the [0, 1] range
x_train = x_train.astype("float32") / 128 - 1
x_test = x_test.astype("float32") / 128 - 1
# Make sure images have shape (28, 28, 1)
#x_train = np.expand_dims(x_train, -1)
#x_test = np.expand_dims(x_test, -1)
print("x_train shape:", x_train.shape)
print(x_train.shape[0], "train samples")
print(x_test.shape[0], "test samples")

# TODO make this more readable
n = y_train.size
y_train_new = np.zeros((n, 10))
for i in range(n):
    y_train_new[i][y_train[i]] = 1
n = y_test.size
y_test_new = np.zeros((n, 10))
for i in range(n):
    y_test_new[i][y_test[i]] = 1
y_train = y_train_new
y_test = y_test_new

if __name__ == '__main__':
    inputs = Input(shape=(28, 28))
    x = Flatten()(inputs)
    x = Dense(128, activation="relu")(x)
    outputs = Dense(10, activation="softmax")(x)
    model = Model(inputs, outputs)

    model.summary()

    opt = Standard(learning_rate=0.01)
    model.compile(loss="mse", optimizer=opt)

    model.fit(x_train, y_train)
    print(model.predict(np.ones((28, 28))))