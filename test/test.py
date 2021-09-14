import numpy as np
from sequential import Sequential


# This tests model.predict on np.ones((28,28))
def prediction_test1():
    model = Sequential()
    pred = model.predict(np.ones((28, 28)))
    expected = np.full(10, ((28 * 28 + 1) ** 2 + (28 * 28 + 1)) * 128 + 1)
    if np.array_equal(pred, expected):
        print('prediction_test1 PASSED!')
    else:
        print('prediction_test1 FAILED!')
        print('Actual:')
        print(pred)
        print('Expected:')
        print(expected)


def prediction_test2():
    model = Sequential()
    pred = model.predict(np.zeros((28, 28)))
    expected = np.full(10, (1 ** 2 + 1) * 128 + 1)
    if np.array_equal(pred, expected):
        print('prediction_test2 PASSED!')
    else:
        print('prediction_test2 FAILED!')
        print('Actual:')
        print(pred)
        print('Expected:')
        print(expected)


def fit_test1():
    model = Sequential()

    # Before fit
    if not np.array_equal(model.w_ij, np.ones((28 * 28 + 1, 128))):
        print('fit_test1 failed')
    if not np.array_equal(model.w_ok, np.ones((128 + 1, 10))):
        print('fit_test1 failed')

    model.fit(np.ones((1, 28, 28)), np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0])[np.newaxis])

    ### After fit

    ### prediction test
    ### first hidden layer

    # o_i
    if not np.array_equal(model.o_i, np.ones(28 * 28 + 1)):
        print('fit_test1 failed: 0')
    # net_j
    if not np.array_equal(model.net_j, np.full(128, 28 * 28 + 1)):
        print('fit_test1 failed: 1')
    # o_j
    if not np.array_equal(model.o_j, np.append(np.full(128, 617010), 1)):
        print('fit_test1 failed: 2')

    ### Output neurons

    # net_k
    if not np.array_equal(model.net_k, np.full(10, 617010 * 128 + 1)):
        print('fit_test1 failed: 3')
    # o_k
    if not np.array_equal(model.o_k, model.net_k):
        print('fit_test1 failed: 4')
        print('model.o_k')
        print(model.o_k)
        print('Expected:')
        print(model.net_k)

    # fit test

    # delta_k
    # net_k correct
    # o_k correct
    # y_train correct
    expected = (lambda x: 2 * x + 1)(model.net_k) * (model.o_k - np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0]))
    if not np.array_equal(model.delta_k, expected):
        print('fit_test1 failed: 5')
        print('Actual:')
        print(model.delta_k)
        print('Expected:')
        print(expected)

    # DeltaW_ok
    # delta_k correct
    expected = 1 * model.delta_k * model.o_j
    if not np.array_equal(model.DeltaW_ok, expected):
        print('fit_test1 failed: 6')
        print('Actual:')
        print(model.DeltaW_ok)
        print('Expected:')
        print(expected)


prediction_test1()
prediction_test2()
fit_test1()
