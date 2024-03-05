import json
import numpy as np
import math
from deepteddy import optimizers, costs
from prettytable import PrettyTable

'''
def import_cupy():
    global np
    np = __import__('cupy')
'''

# Neural Network Object
class Network:

    # initialize the network
    def __init__(self):
        self.layers = []        # each item is a layer object
        self.parameters = []    # each item is a dict w/ 'W' and 'b' matrix for weights and biases, resp. - one item per layer in network
        self.caches = []        # each item is a dict with 'A_prev', 'W', 'b', 'Z' values for the current layer
        self.costs = []         # each item is the cost value for 1 epoch
        
    # add a layer to the network
    def add_layer(self, layer):
        self.layers.append(layer)

    # configure the network's settings - type of cost function, how many inputs, type of optimizer
    def configure_network(self, 
                          input_layer_size, 
                          cost_func=costs.MSE(), 
                          optimizer=optimizers.SGD()): 
        
        self.cost_func = cost_func
        self.optimizer = optimizer
        self.initialize_parameters(input_layer_size)
        self.parameters = self.optimizer.build(self.parameters, self.layers)

    # train the neural network
    def train(self, X, Y, epochs, learning_rate=0.01, minibatch_size=None, verbose=False):
        num_prints = 10 if epochs >= 10 else epochs
        self.optimizer.learning_rate = learning_rate
        m = X.shape[1]  # number of training examples
        if not minibatch_size: minibatch_size = m   # dafault minibatch size is whole training set - batch GD
        
        # for every epoch
        for epoch in range(1, epochs + 1):
            minibatches = self.random_mini_batch(X, Y, minibatch_size=minibatch_size)

            # for each minibatch
            for X_minibatch, Y_minibatch in minibatches:
                AL = self.forward_propagate(X_minibatch)
                cost = self.cost_func.forward_cost(AL, Y_minibatch) + self.layers[-1].regularizer.cost_adjustment(m, self.parameters, self.layers)
                gradients = self.backpropagate(AL, Y_minibatch)  
                self.parameters = self.optimizer.update_step(self.parameters, self.layers, gradients, X) 
                self.costs.append(cost)

            # spit out costs to screen
            if verbose and (epoch % (epochs // num_prints) == 0 or epoch == epochs):
                print(f'Cost of epoch {epoch}: {round(cost.item(), 5)}')

    # initialize the network's parameters according to the specified initializer
    def initialize_parameters(self, input_layer_size):
        layer_sizes = [input_layer_size] + [layer.num_nodes for layer in self.layers] # num. of nodes/neurons in each layer
        self.parameters = [self.layers[layer].initializer(
            layer_sizes[layer + 1], 
            layer_sizes[layer]
            )
            for layer in range(len(layer_sizes) - 1)
        ]

    # split the training examples and labels into randomly permuted minibatches
    def random_mini_batch(self, X, Y, minibatch_size=None):
        m = X.shape[1]  # number of training examples
        permutation = list(np.random.permutation(m))  # random permutation, equal to number of training examples
        minibatches = []  # list to hold minibatches. Each element is a tuple (minibatch_X, minibatch_Y)

        # shuffle the data synchronously
        shuffled_X = X[:, permutation]
        shuffled_Y = Y[:, permutation].reshape((Y.shape[0], m))  # call reshape to make sure the shape is correct

        num_complete_minibatches = math.floor(m / minibatch_size)  # number of minibatches of size minibatch_size
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
    
    # forward propagation
    def forward_propagate(self, A):
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
    
    # backpropagation
    def backpropagate(self, AL, Y):
        gradients = [None] * len(self.layers)   # initialize an array to hold the gradients for each GD step

        # compute derivative of cost wrt output layer's activation.
        dA_prev = self.cost_func.cost_derivative(AL, Y.reshape(AL.shape))  

        # go backward through the network, find dA, dW, and db for all layers
        for layer in reversed(range(len(self.layers))):
            cache = self.caches[layer]
            dA_prev, dW, db = self.layers[layer].layer_backward_prop(dA_prev, cache['A_prev'], cache['W'], cache['b'], cache['Z'])  
            gradients[layer] = {'dW': dW, 'db': db}    # store the gradients of the parameters in gradients array

        return gradients
    
    def predict(self, X):
        return self.forward_propagate(X)

    # network summary
    def network_summary(self, print_table=True):
        num_parameters = [layer['W'].size + layer['b'].size for layer in self.parameters]

        # make a table
        table = PrettyTable(['Layer Type', 'Parameters'])
        for index, layer in enumerate(self.layers):
            table.add_row([type(layer).__name__, num_parameters[index]])

        # print summary
        if print_table:
            print(table)
            print('Total Parameters:', sum(num_parameters))

        return sum(num_parameters)
    
    # write weights and biases to a json file
    def write_parameters(self, name='parameters.json', dir=''):
        json_parameters = [
            {'W': layer['W'].tolist(), 'b': layer['b'].tolist()}
            for layer in [layer for layer in self.parameters]
        ]
        with open(dir + name, 'w') as file:
            json.dump(json_parameters, file)

    # load parameters from a json file
    def read_parameters(self, name='parameters.json', dir=''):
        with open(dir + name, 'r') as file:
            json_parameters = json.load(file)
        self.parameters = [{'W': np.array(layer['W']), 'b': np.array(layer['b'])} for layer in json_parameters]