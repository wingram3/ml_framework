import numpy as np

'''
def import_cupy():
    global np
    np = __import__('cupy')
'''

"""
    Each activation function will be instantiated when creating a layer of neurons.
    The activations have two main parts:
        1 - forward part: for computing the activations (A) of a layer during forward propagation
        2 - backward part: computes the derivative of the cost function w/ resp. to Z
            a - backward function returns dA * dZ; if at the output layer, dA=dA_prev=1 
            (see backpropagate() method of Network class in network.py)
"""

# maps the output to [0, 1] - probability function for binary classification
class Sigmoid:

    def __init__(self, c=1):
        self.c = c

    # 1 / (1 + e^-cz)
    def forward(self, Z):
        return 1 / (1 + np.exp(-self.c * Z))
    
    # c * s(z) * (1 - s(z))
    def backward(self, Z, dA=1):
        s = self.forward(Z)
        dZ = self.c * s * (1 - s)
        return dA * dZ


# recitified linear unit 
class ReLU:

    def __init__(self):
        pass

    # max(0,z)
    def forward(self, Z):
        return np.maximum(0, Z)
    
    # 0 if z <= 0, 1 if z > 0
    def backward(self, Z, dA=1):
        dZ = np.where(Z <= 0, 0, 1)
        return dA * dZ
    

# Softmax - for multi-class classification output layers - use with categorical cross entropy cost function
class Softmax:

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
