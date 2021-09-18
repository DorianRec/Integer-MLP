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
    # For softmax   w_ij 0.02, w_ok 0.015
    # f     g       eta     w_ij    w_ok    correct accuracy    loss    n

    # x^2+x 1+x     0.006   0.02    0.015   30.34   0.38        nan     10000
    # x^2+x 1+x     0.006   0.02    0.015   28.99   0.37        nan     20000
    # x^2+x 1+x     0.01    0.02    0.015   30.42   0.37        nan     10000
    # x^2+x 1+x     0.02    0.02    0.001   27.95   0.37        nan     10000
    # x^2+x 1+x     0.02    0.02    0.015   30.38   0.37        nan     10000
    # x^2+x 1+x     0.02    0.02    0.03    28.10   0.37        nan     10000
    # x^2+x 1+x     0.02    0.05    0.03    26.88   0.38        nan     10000
    # x^2+x 1+x     0.02    0.05    0.05    34.99   0.39        nan     10000
    # x^2+x 1+x     0.02    0.05    0.1     28.10   0.40        nan     10000
    # x^2+x 1+x     0.02    0.1     0.05    28.97   0.40        nan     10000
    # x^2+x 1+x     0.03    0.07    0.07    29.47   0.38        nan     10000
    # x^2+x 1+x     0.05    0.05    0.05    32.30   0.38        nan     10000
    # x^2+x 1+x     0.1     0.05    0.05    29.69   0.37        nan     10000
    # x^2+x 1+x     0.1     0.1     0.1      9.80   0.45        0.23    10000
    # x^2+x 1+x     0.25    0.05    0.05    19.50   0.41        nan     10000

    # x^2+x x^2     0.006   0.02    0.01    11.89   0.66        0.42    10000
    # x^2+x x^2     0.006   0.02    0.015   30.22   0.40        0.19    10000
    # x^2+x x^2     0.006   0.02    0.02    29.09   0.40        0.19    10000
    # x^2+x x^2     0.006   0.025   0.015   28.18   0.40        0.19    10000
    # x^2+x x^2     0.006   0.025   0.02    27.12   0.40        0.19    10000
    # x^2+x x^2     0.01    0.02    0.015   37.1    0.39        0.19    10000
    # x^2+x x^2     0.01    0.02    0.02    24.73   0.41        0.19    10000
    # x^2+x x^2     0.02    0.025   0.015   10.48   0.58        0.35    10000
    # x^2+x x^2     0.02    0.02    0.015    3.42   0.55        0.35    10000

    # x^2+x softmax 0.006   0.02    0.015   49.93   0.31        0.13    10000
    # x^2+x softmax 0.02    0.02    0.015   nan     nan         nan     10000
    w_ij = np.fromfunction(lambda i, j: (-0.07) + 0.14 * ((i + j) % 2), (I_INDEX + 1, J_INDEX))
    w_ijT = np.fromfunction(lambda i, j: (-0.07) + 0.14 * ((i + j) % 2), (J_INDEX, I_INDEX + 1))
    w_ok = np.fromfunction(lambda i, j: (-0.07) + 0.14 * ((i + j) % 2), (O_INDEX + 1, K_INDEX))
    w_okT = np.fromfunction(lambda i, j: (-0.07) + 0.14 * ((i + j) % 2), (K_INDEX, O_INDEX + 1))

    # learning:
    eta = 0.03
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
            # TODO softmax
            # softmax(net_k) - y_o
            #self.delta_k = self.o_k - y_train[x_train_akt]
            # TODO for x^2 approximation
            # dot_product = np.dot(self.net_k, self.net_k)
            # self.delta_k = (self.o_k - y_train[x_train_akt]) * self.net_k * 2 * (dot_product - self.net_k ** 2) / dot_product ** 2
            # self.delta_k = (self.o_k - y_train[x_train_akt]) * 2*self.net_k *(1/dot_product - (self.net_k / dot_product)**2)
            # TODO f(x)=1+x/sum(1+x)
            self.delta_k = (self.o_k - y_train[x_train_akt]) * (np.sum(1 + self.net_k) - (1 + self.net_k)) / (np.sum(1 + self.net_k)) ** 2

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
        # TODO calculate o_k := softmax(net_k)
        #self.o_k = scipy.special.softmax(self.net_k)
        # self.o_k = np.exp(self.o_k) / sum(np.exp(self.o_k))
        # TODO This is the x^2/sum x^2 approximation
        # self.o_k = self.net_k ** 2 / sum(self.net_k ** 2)
        # TODO f(x)=1+x/sum(1+x)
        self.o_k = (self.net_k + 1) / np.sum(self.net_k + 1)

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
