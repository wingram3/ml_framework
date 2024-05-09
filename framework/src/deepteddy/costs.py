import numpy as np


class BinaryCrossEntropy:
    """ Binary cross entropy cost function class. For binary classification. """

    def forward_cost(self, AL: np.ndarray, Y: np.ndarray):
        """ Computes the cost between the last layer acitvations and the labels. """
        m = Y.shape[1]
        return np.squeeze(-1/m * np.sum(np.dot(np.log(AL.T), Y) + np.dot(np.log(1 - AL.T), 1 - Y)))

    def cost_derivative(self, AL: np.ndarray, Y: np.ndarray):
        """ Computes the derivative of the cost. """
        return -Y / AL + (1 - Y) / (1 - AL)


class CategoricalCrossEntropy:
    """ Categorical cross entropy cost function class. For multi-class classification. """

    def forward_cost(self, AL: np.ndarray, Y: np.ndarray):
        """ Computes the cost between the last layer acitvations and the labels. """
        m = Y.shape[1]
        return np.squeeze(-1/m * np.sum(Y * np.log(AL)))

    def cost_derivative(self, AL: np.ndarray, Y: np.ndarray):
        """ Computes the derivative of the cost. """
        return AL - Y


class MSE:
    """ Mean squared error cost function class. For regression tasks. """

    def forward_cost(self, AL: np.ndarray, Y: np.ndarray):
        """ Computes the cost between the last layer acitvations and the labels. """
        m = Y.shape[1]
        return np.squeeze(1/m * np.sum(np.square((Y - AL))))

    def cost_derivative(self, AL: np.ndarray, Y: np.ndarray):
        """ Computes the derivative of the cost. """
        return -2 * (Y - AL)
