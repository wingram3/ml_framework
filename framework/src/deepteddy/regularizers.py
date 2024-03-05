import numpy as np

'''
def import_cupy():
    global np
    np = __import__('cupy')
'''

# Layer weight regularizers:
    # allows for applying penalties to parameters on a per-layer basis to reduce overfitting to training data
    # weight_adjustment() method: for adding a term to the computation of dW during backpropagation through a layer
    # cost_adjustment() method: for adding a term to the forward cost computation 


# L2 Regularization
class L2:

    def __init__(self, lmbda=0):
        self.lmbda = lmbda

    def weight_adjustment(self, m, W):
        return self.lmbda/m * W
    
    def cost_adjustment(self, m, parameters, layers):
        norm = 0
        lmbda_sum = 0
        for layer in range(len(layers)):
            norm += np.sum(np.square(parameters[layer]['W']))
            lmbda_sum += layers[layer].regularizer.lmbda

        return 1/m * (lmbda_sum/len(layers))/2 * norm
    

# L1 Regularization
class L1:

    def __init__(self, lmbda=0):
        self.lmbda = lmbda

    def weight_adjustment(self, m, W):
        return self.lmbda/m * W
    
    def cost_adjustment(self, m, parameters, layers):
        norm = 0
        lmbda_sum = 0
        for layer in range(len(layers)):
            norm += np.sum(np.absolute(parameters[layer]['W']))
            lmbda_sum += layers[layer].regularizer.lmbda

        return 1/m * (lmbda_sum/len(layers))/2 * norm
