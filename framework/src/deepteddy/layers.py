import numpy as np
from deepteddy import activations, regularizers, initializers

'''
def import_cupy():
    global np
    np = __import__('cupy')
'''

# dense layer -> every neuron in the layer is connected to every neuron in the previous layer
class Dense:

    def __init__(self, 
                 num_nodes, 
                 activation=activations.Sigmoid(), 
                 initializer=initializers.he_normal,
                 regularizer=regularizers.L2(lmbda=0.0)):   # a lambda value of zero effectively means there is no regularization on a layer

        self.num_nodes = num_nodes
        self.activation = activation
        self.initializer = initializer   
        self.regularizer = regularizer

    def layer_forward_prop(self, A_prev, W, b):
        Z = np.dot(W, A_prev) + b       
        A = self.activation.forward(Z)  
        return A, Z   
    
    def layer_backward_prop(self, dA, A_prev, W, b, Z):
        m = A_prev.shape[1]     
        dZ = self.activation.backward(Z, dA)    
        dW = 1/m * np.dot(dZ, A_prev.T) + self.regularizer.weight_adjustment(m, W)  
        db = 1/m * np.sum(dZ, axis=1, keepdims=True)   
        dA_prev = np.dot(W.T, dZ)   

        return dA_prev, dW, db
