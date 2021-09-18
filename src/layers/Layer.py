class Layer:
    # Context
    name = ""
    prev = None
    # TODO is nxt always needed ?
    nxt = None
    isOutput = False

    # Variables
    # We do not concatenate the bias here, because there is not always a bias between two layers!
    o_j = None

    delta_j = None

    def get_shape(self):
        pass

    def get_params(self):
        pass

    # x_train.shape = (n, x1, x2)
    # y_train.shape = (n, y1)
    def fit(self, x_train, y_train):
        pass

    def predict(self, x_test):
        pass

    def summary(self):
        assert self.name != ""

        if self.nxt is not None:
            return self.name + "\t\t\t" + str(self.get_shape()) + "\t\t\t" + str(self.get_params()) + "\n" + self.nxt.summary()
        else:
            return self.name + "\t\t\t" + str(self.get_shape()) + "\t\t\t" + str(self.get_params()) + "\n"
