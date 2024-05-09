import json
import numpy as np
import math
from typing import List, Dict, Tuple
from deepteddy import optimizers, costs, layers
from prettytable import PrettyTable


class Network:
    """
    Neural Network (multilayer perceptron) class. Configurable and trainable.

    A Network object is initialized with variables for layers, parameters, caches, and costs.
    Once created, the Network object can have layers added, be configured with optimizers, costs,
    etc., and ultimately trained.

    Attributes:
        layers: each item is a layer object
        parameters: each item is a dict with 'W' and 'b' for weights and biases
        caches: each item is a dict with 'A_prev', 'W', 'b', 'Z' for the current layer
        costs: the cost value for epoch 1
    """
    def __init__(self):
        self.layers = []
        self.parameters = []
        self.caches = []
        self.costs = []


    def add_layer(self, layer: layers.Dense) -> None:
        """ Add a layer to the network object. """
        self.layers.append(layer)


    def configure_network(
        self,
        input_layer_size: int,
        cost_func=costs.MSE(),
        optimizer=optimizers.SGD()) -> None:
        """
        Configure the network's settings - cost func., number of inputs, optimizer.

        Args:
            input_layer_size: the number of inputs into the network
            cost_func: the cost function of the network
            optimizer: the optimization algorithm of the network
        """
        self.cost_func = cost_func
        self.optimizer = optimizer
        self.initialize_parameters(input_layer_size)
        self.parameters = self.optimizer.build(self.parameters, self.layers)


    def train(
        self,
        X,
        Y,
        epochs: int,
        minibatch_size: None,
        learning_rate: float = 0.01,
        verbose: bool = False) -> None:
        """
        Train the Network object.

        Args:
            X: the training inputs, a Numpy array
            Y: the training labels, a Numpy array
            epochs: the number of epochs to train for
            minibatch_size: size of the minibatches (int)
            learning_rate: float representing the learning rate
            verbose: if True, prints cost per epoch to terminal
        """
        num_prints = 10 if epochs >= 10 else epochs
        self.optimizer.learning_rate = learning_rate
        m = X.shape[1]
        if not minibatch_size: minibatch_size = m

        for epoch in range(1, epochs + 1):
            minibatches = self.random_mini_batch(X, Y, minibatch_size=minibatch_size)

            for X_minibatch, Y_minibatch in minibatches:
                AL = self.forward_propagate(X_minibatch)
                cost = self.cost_func.forward_cost(AL, Y_minibatch) + self.layers[-1].regularizer.cost_adjustment(m, self.parameters, self.layers)
                gradients = self.backpropagate(AL, Y_minibatch)
                self.parameters = self.optimizer.update_step(self.parameters, self.layers, gradients, X)
                self.costs.append(cost)

            if verbose and (epoch % (epochs // num_prints) == 0 or epoch == epochs):
                print(f'Cost of epoch {epoch}: {round(cost.item(), 5)}')


    def initialize_parameters(self, input_layer_size: int) -> None:
        """
        Initialize the Network's parameters according to the specified initializer.

        Args:
            input_layer_size: the number of inputs to the Network
        """
        layer_sizes = [input_layer_size] + [layer.num_nodes for layer in self.layers]
        self.parameters = [self.layers[layer].initializer(
            layer_sizes[layer + 1],
            layer_sizes[layer]
            )
            for layer in range(len(layer_sizes) - 1)
        ]


    def random_mini_batch(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        minibatch_size=None) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Split the training examples and labels into randomly permuted minibatches.

        Args:
            X: the training inputs, a Numpy array
            Y: the training labels, a Numpy array
            minibatch_size: size of the minibatches (int)

        Returns:
            A list of the training examples minibatches.
        """
        m = X.shape[1]
        permutation = list(np.random.permutation(m))
        minibatches = []
        shuffled_X = X[:, permutation]
        shuffled_Y = Y[:, permutation].reshape((Y.shape[0], m))

        num_complete_minibatches = math.floor(m / minibatch_size)
        for k in range(0, num_complete_minibatches):
            minibatch_X = shuffled_X[:, k*minibatch_size : (k+1)*minibatch_size]
            minibatch_Y = shuffled_Y[:, k*minibatch_size : (k+1)*minibatch_size]
            minibatch = (minibatch_X, minibatch_Y)
            minibatches.append(minibatch)

        if m % minibatch_size != 0:
            minibatch_X = shuffled_X[:, minibatch_size*num_complete_minibatches : m]
            minibatch_Y = shuffled_Y[:, minibatch_size*num_complete_minibatches : m]
            minibatch = (minibatch_X, minibatch_Y)
            minibatches.append(minibatch)

        return minibatches


    def forward_propagate(self, A: np.ndarray) -> np.ndarray:
        """
        When called, performs forward propagation through the Network.

        Args:
            A: a Numpy array of neuron activations for one layer

        Returns:
            The activations of the next layer of neurons.
        """
        self.caches = []

        for layer in range(len(self.layers)):
            A_prev = A
            A, Z = self.layers[layer].layer_forward_prop(A_prev, self.parameters[layer]['W'], self.parameters[layer]['b'])
            self.caches.append({
                'A_prev': A_prev,
                'W': self.parameters[layer]['W'],
                'b': self.parameters[layer]['b'],
                'Z': Z
            })

        return A


    def backpropagate(self, AL: np.ndarray, Y: np.ndarray) -> List[Dict[str, np.ndarray]]:
        """
        When called, performs backpropagation through the Network.

        Args:
            AL: a Numpy array of the output layer's activations

        Returns:
            A list of dictionaries of weights and biases
        """
        gradients = [] * len(self.layers)
        dA_prev = self.cost_func.cost_derivative(AL, Y.reshape(AL.shape))
        for layer in reversed(range(len(self.layers))):
            cache = self.caches[layer]
            dA_prev, dW, db = self.layers[layer].layer_backward_prop(dA_prev, cache['A_prev'], cache['W'], cache['b'], cache['Z'])
            gradients[layer] = {'dW': dW, 'db': db}
        return gradients


    def predict(self, X) -> np.ndarray:
        """ Predictions of the Network given inputs X """
        return self.forward_propagate(X)


    def network_summary(self, print_table: bool = True) -> int:
        """
        Print the Network's performace summary.

        Args:
            print_table: if True, prints a table with info about the network

        Returns:
            The number of parameters in the Network.
        """
        num_parameters = [layer['W'].size + layer['b'].size for layer in self.parameters]
        table = PrettyTable(['Layer Type', 'Parameters'])
        for index, layer in enumerate(self.layers):
            table.add_row([type(layer).__name__, num_parameters[index]])

        if print_table:
            print(table)
            print('Total Parameters:', sum(num_parameters))

        return sum(num_parameters)


    def write_parameters(self, name: str = 'parameters.json', dir: str = '') -> None:
        """
        Write the weights and biases to a json file.

        Args:
            name: name of the output json file containing parameters
            dir: the directory to contain the output json file
        """
        json_parameters = [
            {'W': layer['W'].tolist(), 'b': layer['b'].tolist()}
            for layer in [layer for layer in self.parameters]
        ]
        with open(dir + name, 'w') as file:
            json.dump(json_parameters, file)


    def read_parameters(self, name='parameters.json', dir=''):
        """
        Load parameters from a json file.

        Args:
            name: the input json file containing the paramneters
            dir: the directory containing the json file
        """
        with open(dir + name, 'r') as file:
            json_parameters = json.load(file)
        self.parameters = [{'W': np.array(layer['W']), 'b': np.array(layer['b'])} for layer in json_parameters]
