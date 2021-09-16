from tensorflow import keras
import numpy as np

from Integer_Sequential import Integer_Sequential
from Sequential import Sequential

# Model / data parameters
num_classes = 10
input_shape = (28, 28, 1)

# the data, split between train and test sets
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
# TODO remove (test purposes)
(x_train, y_train), (x_test, y_test) = (x_train[0:10001], y_train[0:10001]), (x_test, y_test)
# Scale images to the [0, 1] range
#x_train = x_train.astype("float32") / 128 - 1
#x_test = x_test.astype("float32") / 128 - 1
# Make sure images have shape (28, 28, 1)
#x_train = np.expand_dims(x_train, -1)
#x_test = np.expand_dims(x_test, -1)
print("x_train shape:", x_train.shape)
print(x_train.shape[0], "train samples")
print(x_test.shape[0], "test samples")

### Build the model

#model = Sequential()
model = Integer_Sequential()

### set up data

y_test_old = y_test

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

### Train the model

model.fit(x_train, y_train)

### Evaluate the model

score = model.evaluate(x_test, y_test, y_test_old)
print("Test loss:", score[0])
# TODO is accuracy = average quadratic error ?
print("Test accuracy:", score[1])

### Model prediction

# pred = model.predict(x_test[0:1])
