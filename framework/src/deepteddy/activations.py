import numpy as np


class Sigmoid:
    """
    Sigmoid nonlinearity class.

    Maps the output to [0, 1].
    Probability function for binary classification.
    """
    def __init__(self, c=1):
        self.c = c

    def forward(self, Z):
        return 1 / (1 + np.exp(-self.c * Z))

    def backward(self, Z, dA=1):
        s = self.forward(Z)
        dZ = self.c * s * (1 - s)
        return dA * dZ


class ReLU:
    """ Rectified linear unit (ReLU) nonlinearity class. """
    def __init__(self):
        pass

    def forward(self, Z):
        return np.maximum(0, Z)

    def backward(self, Z, dA=1):
        dZ = np.where(Z <= 0, 0, 1)
        return dA * dZ


class Softmax:
    """
    Softmax nonlinearity class.

    For multi-class classification output layers.
    Use with categorical cross entropy cost function.
    """
    def __init__(self):
        pass

    def forward(self, Z):
        e = np.exp(Z)
        return e / np.sum(e, axis=1, keepdims=True)

    def backward(self, Z, dA=None):
        shape = Z.shape
        s = self.forward(Z)
        diag = s.reshape(shape[0], -1, 1) * np.diag(np.ones(shape[1]))
        outer = np.matmul(s.reshape(shape[0], -1, 1), s.reshape(shape[0], 1, -1))
        jacobian = diag - outer

        if dA is None: jacobian
        else: return np.matmul(dA.reshape(shape[0], 1, -1), jacobian).reshape(shape)
