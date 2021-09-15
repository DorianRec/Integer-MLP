import numpy as np
import scipy


class Sequential:
    # We used:
    # NN Formulars: https://de.wikipedia.org/wiki/Backpropagation
    # softmax = scipy.special.softmax
    # softmax derivative: partial * E / partial w_ij = o_i * (softmax(net_j) - y_j)
    # See here: https://stats.stackexchange.com/questions/235528/backpropagation-with-softmax-cross-entropy

    # i = 0, ..., 28*28+1 -1
    # j = 0, ..., 128 -1
    # o = 0, ..., 128+1 -1
    # k = 0, ..., 10 -1

    # indices
    i_index = 28 * 28
    j_index = 128
    o_index = 128
    k_index = 10

    # neural net
    bias_1 = 1
    bias_2 = 1
    net_j = np.empty(j_index)  # neurons j=0,..,127
    net_k = np.empty(k_index)  # neurons k=0+128,...,9+128
    o_i = np.empty(i_index + 1)  # input layer
    o_j = np.empty(j_index + 1)[np.newaxis]  # neurons j
    o_k = np.empty(k_index)  # output of output layer
    w_ij = np.fromfunction(lambda i, j: (-0.02) + 0.04 * ((i + j) % 2), (i_index + 1, j_index))
    w_ijT = np.fromfunction(lambda i, j: (-0.02) + 0.04 * ((i + j) % 2), (j_index, i_index + 1))
    w_ok = np.fromfunction(lambda i, j: (-0.015) + 0.03 * ((i + j) % 2), (o_index + 1, k_index))
    w_okT = np.fromfunction(lambda i, j: (-0.015) + 0.03 * ((i + j) % 2), (k_index, o_index + 1))

    # learning
    eta = 0.002  # learning rate
    delta_j = np.empty(j_index)
    delta_k = np.empty(k_index)[np.newaxis]
    DeltaW_ij = np.empty((i_index + 1, j_index))
    DeltaW_ok = np.empty((o_index + 1, k_index))

    def fit(self, x_train, y_train):
        (n, x1, x2) = x_train.shape

        for x_train_akt in range(n):
            print('Iteration: ' + str(x_train_akt))
            x_i = np.append(x_train[x_train_akt].flatten(), 1)

            # calculate delta_k
            prediction = (self.predict(x_train[x_train_akt]))
            # TODO (lambda x: 2 * x + 1)(self.net_k) should be wrong
            # TODO softmax derivative
            # self.delta_k = (prediction - y_train[x_train_akt])
            # softmax(net_j) - y_i
            self.delta_k = self.o_k - y_train[x_train_akt]
            # calculate DeltaW_ok
            # 129x10 = 1 * (129,) (10,)
            # self.DeltaW_ok = -self.eta * np.matmul(self.o_j[np.newaxis].T, self.delta_k[np.newaxis])
            # o_i * (softmax(net_j) - y_j)
            self.DeltaW_ok = -self.eta * self.o_j[np.newaxis].T * self.delta_k.T[np.newaxis]
            # calculate delta_j
            for j in range(self.j_index):
                self.delta_j[j] = (lambda x: 2 * x + 1)(self.net_j[j]) * np.dot(self.delta_k, self.w_ok[j])
            # calculate DeltaW_ij
            # 28*28+1x128 = 1 * (128,)(28*28+1)
            self.DeltaW_ij = -self.eta * np.matmul(x_i[np.newaxis].T, self.delta_j[np.newaxis])

            # Update weights
            self.w_ij += self.DeltaW_ij
            self.w_ijT = self.w_ij.T
            self.w_ok += self.DeltaW_ok
            self.w_okT = self.w_ok.T

            # TODO Move to correct place
            # self.o_k = self.o_k / np.sum(self.o_k)

            # print error
            # sqrt( sum (o_k[k] - y_train[k])^2 )

            print('o_k: ' + str(self.o_k))
            print('y_train: ' + str(y_train[x_train_akt]))
            tmp = y_train[x_train_akt] - self.o_k
            print('Quadratic error: ' + str(0.5 * np.dot(tmp, tmp)))

    # Takes shape (28, 28)
    def predict(self, x_test):
        assert x_test.shape == (28, 28)

        # calculate o_i
        self.o_i = np.append(x_test.flatten(), [self.bias_1])

        # calculate net_j
        for j in range(self.j_index):
            self.net_j[j] = np.dot(self.o_i, self.w_ijT[j])

        # calculate o_j
        # Apply x^2 + x to net_j
        self.o_j = np.append(list(map(lambda x: x ** 2 + x, self.net_j)), [self.bias_2])

        # calculate net_k
        for k in range(self.k_index):
            self.net_k[k] = np.dot(self.o_j, self.w_okT[k])
        # calculate o_k
        # TODO add softmax
        self.o_k = scipy.special.softmax(self.net_k)
        # self.o_k = self.net_k

        return self.o_k

    def evaluate(self, x_test, y_test, y_test_old):
        (n, x1, x2) = x_test.shape
        (n1, y1) = y_test.shape
        assert n == n1
        # This contains the quadratic error: 1/2 * sum (y_i - o_k)^2
        error_vector = np.empty(n)
        # This contains the guesses
        guess_vector = np.empty(n)
        for i in range(n):
            o_k = self.predict(x_test[i])
            diff = y_test[i] - o_k
            error_vector[i] = 1 / 2 * np.dot(diff, diff)
            guess_vector[i] = (np.argmax(o_k) == y_test_old[i])
        correct_guesses = np.count_nonzero(guess_vector)

        print('Evaluation:')
        print('Test dataset: ' + str(n))
        print('Correct guesses: ' + str(correct_guesses) + ' / ' + str(n) + ', ' + str(correct_guesses/n*100) + '%.')
        print('Average error: ' + str(np.average(error_vector)))
        print('Maximum error: ' + str(np.max(error_vector)))
        print('Minimum error: ' + str(np.min(error_vector)))

        return np.sum(error_vector)
