import numpy as np
from typing import Tuple
from deepteddy import activations, regularizers, initializers


class Dense:
    """
    Dense layer class

    Every neuron in a Dense layer object is connected to every neuron in the
    previous layer, or to each input in the input layer.

    Attributes:
        num_nodes: the number of neurons in the layer
        activation: the nonlinearity (activation function) for the layer
        initializer: the method used to initialize the layer's weights and biases
        regularizer: the regularizer (L1 or L2) of the layer
    """
    def __init__(self,
                 num_nodes,
                 activation=activations.Sigmoid(),
                 initializer=initializers.he_normal,
                 regularizer=regularizers.L2(lmbda=0.0)):

        self.num_nodes = num_nodes
        self.activation = activation
        self.initializer = initializer
        self.regularizer = regularizer


    def layer_forward_prop(self, A_prev, W, b) -> Tuple[np.ndarray, np.ndarray]:
        """
        Performs forward propagation through the layer.

        Args:
            A_prev: the neuron activations of the previous layer
            W: the weights
            b: the biases

        Returns:
            Z: dot product of the prev. layer's activations and current layer's weights plus bias
            A: Z fed through nonlinearity
        """
        Z = np.dot(W, A_prev) + b
        A = self.activation.forward(Z)
        return A, Z


    def layer_backward_prop(self, dA, A_prev, W, b, Z) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Performs backpropagation through the layer.

        Args:
            dA: gradients of layer activations
            A_prev: the neuron activations of the previous layer
            W: the weights
            b: the biases
            Z: dot product of the prev. layer's activations and current layer's weights plus bias

        Returns:
            dA_prev: gradients of the previous layer's activations
            dW: gradients of the weights w.r.t the cost function
            db: gradients of the biases w.r.t the cost function
        """
        m = A_prev.shape[1]
        dZ = self.activation.backward(Z, dA)
        dW = 1/m * np.dot(dZ, A_prev.T) + self.regularizer.weight_adjustment(m, W)
        db = 1/m * np.sum(dZ, axis=1, keepdims=True)
        dA_prev = np.dot(W.T, dZ)
        return dA_prev, dW, db
