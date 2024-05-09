import numpy as np
from typing import List


class L2:
    """
    L2 regularization class.

    Allows for applying penalties to parameters on a per-layer basis to reduce overfitting.

    Attributes:
        lmbda: regularization parameter
    """
    def __init__(self, lmbda: float = 0.0):
        self.lmbda = lmbda

    def weight_adjustment(self, m: int, W: np.ndarray) -> np.ndarray:
        """ Adds a term to the computation of dW during backpropagation through a layer. """
        return self.lmbda/m * W

    def cost_adjustment(self, m: int, parameters: List, layers: List) -> float:
        """ Adds a term to the forward cost computation. """
        norm = 0
        lmbda_sum = 0
        for layer in range(len(layers)):
            norm += np.sum(np.square(parameters[layer]['W']))
            lmbda_sum += layers[layer].regularizer.lmbda
        return 1/m * (lmbda_sum/len(layers))/2 * norm


class L1:
    """
    L1 regularization class.

    Allows for applying penalties to parameters on a per-layer basis to reduce overfitting

    Attributes:
        lmbda: regularization parameter
    """
    def __init__(self, lmbda: float = 0.0):
        self.lmbda = lmbda

    def weight_adjustment(self, m: int, W: np.ndarray) -> np.ndarray:
        """ Adds a term to the computation of dW during backpropagation through a layer. """
        return self.lmbda/m * W

    def cost_adjustment(self, m: int, parameters: List, layers: List) -> float:
        """ Adds a term to the forward cost computation. """
        norm = 0
        lmbda_sum = 0
        for layer in range(len(layers)):
            norm += np.sum(np.absolute(parameters[layer]['W']))
            lmbda_sum += layers[layer].regularizer.lmbda
        return 1/m * (lmbda_sum/len(layers))/2 * norm
