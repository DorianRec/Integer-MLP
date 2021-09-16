import numpy as np
import scipy
from scipy import special


class Sequential:
    # We used:
    # NN Formulars: https://de.wikipedia.org/wiki/Backpropagation
    # approximation of relu(x):=max(o,x): x^2 + x
    # softmax(x) := e^(x - max(x)) / sum(e^(x - max(x))
    # Implemented: softmax = scipy.special.softmax
    # Implemented: softmax derivative: partial * E / partial w_ij = o_i * (softmax(net_j) - y_j)
    # (See here) https://stats.stackexchange.com/questions/235528/backpropagation-with-softmax-cross-entropy

    # indices
    I_INDEX = 28 * 28
    J_INDEX = 128
    O_INDEX = 128
    K_INDEX = 10

    # neural net
    BIAS_1 = 1
    BIAS_2 = 1
    net_j = np.empty(J_INDEX)  # neurons j=0,..,127
    net_k = np.empty(K_INDEX)  # neurons k=0+128,...,9+128
    o_i = np.empty(I_INDEX + 1)  # input layer
    o_j = np.empty(J_INDEX + 1)[np.newaxis]  # neurons j
    o_k = np.empty(K_INDEX)  # output of output layer
    w_ij = np.fromfunction(lambda i, j: (-0.02) + 0.04 * ((i + j) % 2), (I_INDEX + 1, J_INDEX))
    w_ijT = np.fromfunction(lambda i, j: (-0.02) + 0.04 * ((i + j) % 2), (J_INDEX, I_INDEX + 1))
    w_ok = np.fromfunction(lambda i, j: (-0.015) + 0.03 * ((i + j) % 2), (O_INDEX + 1, K_INDEX))
    w_okT = np.fromfunction(lambda i, j: (-0.015) + 0.03 * ((i + j) % 2), (K_INDEX, O_INDEX + 1))

    # learning
    eta = 0.002  # learning rate
    delta_j = np.empty(J_INDEX)
    delta_k = np.empty(K_INDEX)[np.newaxis]
    DeltaW_ij = np.empty((I_INDEX + 1, J_INDEX))
    DeltaW_ok = np.empty((O_INDEX + 1, K_INDEX))

    def fit(self, x_train, y_train):
        (n, x1, x2) = x_train.shape

        for x_train_akt in range(n):
            print('Iteration: ' + str(x_train_akt))
            x_i = np.append(x_train[x_train_akt].flatten(), 1)

            # First call the prediction:
            prediction = (self.predict(x_train[x_train_akt]))

            # calculate delta_k
            # softmax(net_k) - y_o
            self.delta_k = self.o_k - y_train[x_train_akt]
            # calculate DeltaW_ok
            # DeltaW_ok = o_o * (softmax(net_k) - y_k)
            self.DeltaW_ok = -self.eta * np.matmul(self.o_j[np.newaxis].T, self.delta_k[np.newaxis])
            # calculate delta_j
            for j in range(self.J_INDEX):
                self.delta_j[j] = (lambda x: 2 * x + 1)(self.net_j[j]) * np.dot(self.delta_k, self.w_ok[j])
            # calculate DeltaW_ij
            self.DeltaW_ij = -self.eta * np.matmul(x_i[np.newaxis].T, self.delta_j[np.newaxis])

            # Update weights
            self.w_ij += self.DeltaW_ij
            self.w_ijT = self.w_ij.T
            self.w_ok += self.DeltaW_ok
            self.w_okT = self.w_ok.T

            print('o_k: ' + str(self.o_k))
            print('y_train: ' + str(y_train[x_train_akt]))
            tmp = y_train[x_train_akt] - self.o_k
            print('Quadratic error: ' + str(0.5 * np.dot(tmp, tmp)))

    # Takes shape (28, 28)
    def predict(self, x_test):
        assert x_test.shape == (28, 28)

        # calculate o_i
        self.o_i = np.append(x_test.flatten(), [self.BIAS_1])

        # calculate net_j
        for j in range(self.J_INDEX):
            self.net_j[j] = np.dot(self.o_i, self.w_ijT[j])

        # calculate o_j
        # Apply x^2 + x to net_j
        self.o_j = np.append(list(map(lambda x: x ** 2 + x, self.net_j)), [self.BIAS_2])

        # calculate net_k
        for k in range(self.K_INDEX):
            self.net_k[k] = np.dot(self.o_j, self.w_okT[k])
        # calculate o_k
        self.o_k = scipy.special.softmax(self.net_k)

        return self.o_k

    def evaluate(self, x_test, y_test, y_test_old):
        (n, x1, x2) = x_test.shape
        (n1, y1) = y_test.shape
        assert n == n1
        # This contains the quadratic error: 1/2 * sum (y_i - o_k)^2
        error_vector = np.empty(n)
        # This contains the loss
        loss_vector = np.empty(n)
        # This contains the guesses
        guess_vector = np.empty(n)
        for i in range(n):
            o_k = self.predict(x_test[i])
            diff = y_test[i] - o_k
            error_vector[i] = 1 / 2 * np.dot(diff, diff)
            loss_vector[i] = -np.mean(y_test[i] * np.log(o_k.T + 1e-8))
            guess_vector[i] = (np.argmax(o_k) == y_test_old[i])
        correct_guesses = np.count_nonzero(guess_vector)

        print('Evaluation:')
        print('Test dataset: ' + str(n))
        print(
            'Correct guesses: ' + str(correct_guesses) + ' / ' + str(n) + ', ' + str(correct_guesses / n * 100) + '%.')

        return np.average(loss_vector), np.average(error_vector)
