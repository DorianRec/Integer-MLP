import numpy as np


class ReLu:
    def activate(self, x):
        return np.maximum(x, 0)

    def derivative(self, net_j, precomputed_j):
        output = np.empty(net_j.shape[0])
        for j in range(output.shape[0]):
            if net_j[j] < 0:
                output[j] = 0
            else:
                # net_j[j] >= 1:
                # Mathematically, this is not correct for net_j[j] = 0.
                output[j] = 1
        return output
