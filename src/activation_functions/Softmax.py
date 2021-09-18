import scipy
from scipy import special


class Softmax:
    def activate(self, x):
        return scipy.special.softmax(x)

    # Outputs partial softmax(net_j, )/partial net_j
    def derivative(self, net_j, precomputed_j):
        # See https://en.wikipedia.org/wiki/Softmax_function
        # partial softmax(net_j, i) / partial net_j[k] = softmax(net_j, i) (kronecker_delta(i,j) - softmax(net_j, k)
        return precomputed_j * (1 - precomputed_j)

    def kronecker_delta(self, i, j):
        if i == j:
            return 1
        else:
            return 0
