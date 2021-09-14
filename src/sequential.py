import numpy as np


class Sequential:
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
    # w_ij = np.full((i_index + 1, j_index), 1/28*28)
    w_ij = np.fromfunction(lambda i, j: (-0.25) + 0.5 * ((i + j) % 2), (i_index + 1, j_index))
    # w_ijT = np.full((j_index, i_index + 1), 1/28*28)  # Transposed w_ij
    w_ijT = np.fromfunction(lambda i, j: (-0.25) + 0.5 * ((i + j) % 2), (j_index, i_index + 1))
    # w_ok = np.full((o_index + 1, k_index), 1/10)
    w_ok = np.fromfunction(lambda i, j: (-0.25) + 0.5 * ((i + j) % 2), (o_index + 1, k_index))
    # w_okT = np.full((k_index, o_index + 1), 1/10)  # Transposed w_ok
    w_okT = np.fromfunction(lambda i, j: (-0.25) + 0.5 * ((i + j) % 2), (k_index, o_index + 1))

    # learning
    eta = 1  # learning rate
    delta_j = np.empty(j_index)
    delta_k = np.empty(k_index)[np.newaxis]
    DeltaW_ij = np.empty((i_index + 1, j_index))
    DeltaW_ok = np.empty((o_index + 1, k_index))

    def fit(self, x_train, y_train):
        (n, x1, x2) = x_train.shape

        for x_train_akt in range(n):
            x_i = np.append(x_train[x_train_akt].flatten(), 1)

            # calculate delta_k
            prediction = (self.predict(x_train[x_train_akt]))
            self.delta_k = (lambda x: 2 * x + 1)(self.net_k) * (prediction - y_train[x_train_akt])
            # calculate DeltaW_ok
            # 129x10 = 1 * (129,) (10,)
            self.DeltaW_ok = -self.eta * np.matmul(self.o_j[np.newaxis].T, self.delta_k[np.newaxis])
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

            # TODO is this necessary ?
            print('norm')
            print(np.linalg.norm(self.o_k))
            self.o_k = self.o_k / np.linalg.norm(self.o_k)

            # print error
            # sqrt( sum (o_k[k] - y_train[k])^2 )

            tmp = self.o_k - y_train[x_train_akt]
            print('o_k')
            print(self.o_k)
            print('y_train')
            print(y_train[x_train_akt])
            print('error')
            print(np.sqrt(np.dot(tmp, tmp)))

        # TODO update transposed weights.

    # Takes shape (28, 28)
    def predict(self, x_test):

        assert x_test.shape == (28, 28)

        # calculate o_i
        self.o_i = np.append(x_test.flatten(), [self.bias_1])

        # calculate net_j
        for j in range(self.j_index):
            self.net_j[j] = np.dot(self.o_i, self.w_ijT[j])

        # calculate o_j
        self.o_j = np.append(list(map(lambda x: x ** 2 + x, self.net_j)), [self.bias_2])

        # calculate net_k
        for k in range(self.k_index):
            self.net_k[k] = np.dot(self.o_j, self.w_okT[k])
        # calculate o_k
        self.o_k = self.net_k

        return self.o_k
