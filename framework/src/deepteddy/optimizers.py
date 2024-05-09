import numpy as np
from typing import List


class SGD:
    """
    Stochastic Gradient Descent (SGD) class.

    For each step of optimization, updates Network parameters in the
    direction that will decrease the cost the most.

    Attributes:
        learning_rate: determines step size during each optimization step
    """
    def __init__(self, learning_rate: float = 0.001):
        self.learning_rate = learning_rate

    def build(self, parameters: List, layers: List) -> List:
        """ No modification to parameters is needed for SGD, simply return parameters """
        return parameters

    def update_step(
        self,
        parameters: List,
        layers: List,
        gradients: List,
        X: np.ndarray) -> List:
        """ Update the weights and biases according to the learning rate """
        for layer in range(len(layers)):
            m = X.shape[1]
            parameters[layer]['W'] -= self.learning_rate * gradients[layer]['dW']
            parameters[layer]['b'] -= self.learning_rate * gradients[layer]['db']
        return parameters


class RMSProp:
    """
    Root mean squared propagation (RMSProp) class.

    Essentially penalizes the learning rate as the cost function
    nears local minima.

    Attributes:
        learning_rate: determines step size during each optimization step
        beta: hyperparameter to give more or less weight to previous gradients
        epsilon: prevents division by zero errors
    """
    def __init__(self, learning_rate: float = 0.001, beta: float = 0.9, epsilon: float = 1e-8):
        self.learning_rate = learning_rate
        self.beta = beta
        self.epsilon = epsilon

    def build(self, parameters: List, layers: List) -> List:
        """ Initialize learning rate scalers. """
        for layer in range(len(layers)):
            parameters[layer]['SdW'] = np.zeros(parameters[layer]['W'].shape)
            parameters[layer]['Sdb'] = np.zeros(parameters[layer]['b'].shape)
        return parameters

    def update_step(
        self,
        parameters: List,
        layers: List,
        gradients: List,
        X: np.ndarray) -> List:
        """ Update parameters, dividing the learning rate by the square root of the scaler, plus epsilon. """
        for layer in range(len(layers)):
            parameters[layer]['SdW'] = self.beta*parameters[layer]['SdW'] * (1 - self.beta)*np.square(gradients[layer]['dW'])
            parameters[layer]['Sdb'] = self.beta*parameters[layer]['Sdb'] * (1 - self.beta)*np.square(gradients[layer]['db'])
            parameters[layer]['W'] -= self.learning_rate * parameters[layer]['SdW'] / (np.sqrt(parameters[layer]['SdW']) + self.epsilon)
            parameters[layer]['b'] -= self.learning_rate * parameters[layer]['Sdb'] / (np.sqrt(parameters[layer]['Sdb']) + self.epsilon)
        return parameters


class Adam:
    """
    Adaptive moment estimation (Adam) class.

    Combines the benefits of momentum with the benefits of RMSProp.

    Attributes:
        learning_rate: determines step size during each optimization step
        beta1: hyperparameter to give more or less weight to previous gradients
        beta2: hyperparameter to give more or less weight to previous gradients squared
        epsilon: prevents division by zero errors
    """
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon

    def build(self, parameters, layers):
        """ Initialize weight velocities, bias velocites, lr scalers for weights, lr scalers for biases. """
        for layer in range(len(layers)):
            parameters[layer]['VdW'] = np.zeros(parameters[layer]['W'].shape)  # initiliaze weight velocities
            parameters[layer]['Vdb'] = np.zeros(parameters[layer]['b'].shape)  # initialize bias velocities
            parameters[layer]['SdW'] = np.zeros(parameters[layer]['W'].shape)  # initialize leanring scaler for weights
            parameters[layer]['Sdb'] = np.zeros(parameters[layer]['b'].shape)  # initialize learning scaler for biases
        return parameters

    def update_step(self, parameters, layers, gradients, X):
        """ Update parameters, dividing the learning rate by the square root of the scaler, plus epsilon. """
        for layer in range(len(layers)):
            parameters[layer]['VdW'] = self.beta1*parameters[layer]['VdW'] + (1 - self.beta1)*gradients[layer]['dW']
            parameters[layer]['Vdb'] = self.beta1*parameters[layer]['Vdb'] + (1 - self.beta1)*gradients[layer]['db']
            parameters[layer]['SdW'] = self.beta2*parameters[layer]['SdW'] + (1 - self.beta2)*np.square(gradients[layer]['dW'])
            parameters[layer]['Sdb'] = self.beta2*parameters[layer]['Sdb'] + (1 - self.beta2)*np.square(gradients[layer]['db'])
            parameters[layer]['W'] -= self.learning_rate * parameters[layer]['VdW'] / (np.sqrt(parameters[layer]['SdW']) + self.epsilon)
            parameters[layer]['b'] -= self.learning_rate * parameters[layer]['Vdb'] / (np.sqrt(parameters[layer]['Sdb']) + self.epsilon)
        return parameters
