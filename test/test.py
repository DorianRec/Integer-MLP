import numpy as np
from Sequential import Sequential

# TODO This tests are no longer working!

# This tests model.predict on np.ones((28,28))
def prediction_test1():
    # model
    model = Sequential()
    model.w_ij = np.ones((model.I_INDEX + 1, model.J_INDEX))
    model.w_ijT = np.ones((model.J_INDEX, model.I_INDEX + 1))
    model.w_ok = np.ones((model.O_INDEX + 1, model.K_INDEX))
    model.w_okT = np.ones((model.K_INDEX, model.O_INDEX + 1))
    # learning
    model.eta = 1

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
        quit()


def prediction_test2():
    # model
    model = Sequential()
    model.w_ij = np.ones((model.I_INDEX + 1, model.J_INDEX))
    model.w_ijT = np.ones((model.J_INDEX, model.I_INDEX + 1))
    model.w_ok = np.ones((model.O_INDEX + 1, model.K_INDEX))
    model.w_okT = np.ones((model.K_INDEX, model.O_INDEX + 1))
    # learning
    model.eta = 1

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
        quit()


def fit_test1():
    # model
    model = Sequential()
    # need two different copies (since it is one object)
    old_w_ij = np.ones((model.I_INDEX + 1, model.J_INDEX))
    model.w_ij = np.ones((model.I_INDEX + 1, model.J_INDEX))
    model.w_ijT = np.ones((model.J_INDEX, model.I_INDEX + 1))
    # need two different copies (since it is one object)
    old_w_ok = np.ones((model.O_INDEX + 1, model.K_INDEX))
    model.w_ok = np.ones((model.O_INDEX + 1, model.K_INDEX))
    model.w_okT = np.ones((model.K_INDEX, model.O_INDEX + 1))
    # learning
    model.eta = 1

    # Before fit
    if not np.array_equal(model.w_ij, np.ones((28 * 28 + 1, 128))):
        print('fit_test1 failed: 0')
        quit()
    if not np.array_equal(model.w_ok, np.ones((128 + 1, 10))):
        print('fit_test1 failed: 1')
        quit()

    model.fit(np.ones((1, 28, 28)), np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0])[np.newaxis])

    ### After fit

    ### prediction test
    ### first hidden layer

    # o_i
    if not np.array_equal(model.o_i, np.ones(28 * 28 + 1)):
        print('fit_test1 failed: 2')
        quit()
    # net_j
    if not np.array_equal(model.net_j, np.full(128, 28 * 28 + 1)):
        print('fit_test1 failed: 3')
        quit()
    # o_j
    if not np.array_equal(model.o_j, np.append(np.full(128, 617010), 1)):
        print('fit_test1 failed: 4')
        quit()

    ### Output neurons

    # net_k
    if not np.array_equal(model.net_k, np.full(10, 617010 * 128 + 1)):
        print('fit_test1 failed: 5')
        quit()
    # o_k
    if not np.array_equal(model.o_k, model.net_k):
        print('fit_test1 failed: 6')
        print('model.o_k')
        print(model.o_k)
        print('Expected:')
        print(model.net_k)
        quit()

    # fit test

    # delta_k
    # net_k correct
    # o_k correct
    # y_train correct
    expected_7 = (lambda x: 2 * x + 1)(model.net_k) * (model.o_k - np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0]))
    if not np.array_equal(model.delta_k, expected_7):
        print('fit_test1 failed: 7')
        print('Actual:')
        print(model.delta_k)
        print('Expected:')
        print(expected_7)
        quit()

    # DeltaW_ok - Check DeltaW_ok[0][1] only
    # delta_k correct
    expected_8 = -model.eta * model.delta_k[1] * model.o_j[0] # j=o
    if not np.array_equal(model.DeltaW_ok[0][1], expected_8):
        print('fit_test1 failed: 8')
        print('Actual:')
        print(model.DeltaW_ok[0][1])
        print('Expected:')
        print(expected_8)
        quit()

    # delta_j -- Check delta_j[0] only
    # net_j correct
    # delta_k correct
    # DeltaW_jk = DeltaW_ok correct
    expected_9 = (lambda x: 2 * x + 1)(model.net_j[0]) * np.dot(model.delta_k, old_w_ok[0])
    #self.delta_j[j] = (lambda x: 2 * x + 1)(self.net_j[j]) * np.dot(self.delta_k, self.w_ok[j])
    if not np.array_equal(model.delta_j[0], expected_9):
        print('fit_test1 failed: 9')
        print('Actual:')
        print(model.delta_j[0])
        #print('model.net_j[0]')
        #print(model.net_j[0])
        #print('model.delta_k')
        #print(model.delta_k)

        print('Expected:')
        #print('model.net_j[0]')
        #print(model.net_j[0])
        #print('model.delta_k')
        #print(model.delta_k)
        print('old_w_ok[0]')
        print(old_w_ok[0])

        print(expected_9)
        quit()

    # DeltaW_ij

    # w_ij
    if not np.array_equal(model.w_ij[4][2], model.w_ijT[2][4]):
        print('fit_test1 failed: Transposition failed')
        quit()

    # w_ok
    if not np.array_equal(model.w_ok[4][2], model.w_okT[2][4]):
        print('fit_test1 failed: Transposition failed')
        quit()

    print('fit_test1 PASSED!')


prediction_test1()
prediction_test2()
fit_test1()
