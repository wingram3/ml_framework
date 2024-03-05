import numpy as np

'''
def import_cupy():
    global np
    np = __import__('cupy')
'''

"""
Each cost function class has two methods:
    - forward_cost: computes the cost between the last layer acitvations and the labels (after forward propagation)
    - cost_derivative: computes the derivative of the cost
"""

# for binary classification
class BinaryCrossEntropy:

    def forward_cost(self, AL, Y):
        m = Y.shape[1]
        return np.squeeze(-1/m * np.sum(np.dot(np.log(AL.T), Y) + np.dot(np.log(1 - AL.T), 1 - Y)))

    def cost_derivative(self, AL, Y):
        return -Y / AL + (1 - Y) / (1 - AL)


# for multiclass classifcation
class CategoricalCrossEntropy:

    # (-1 / m * sum(Yln(A)))
    def forward_cost(self, AL, Y):
        m = Y.shape[1]
        return np.squeeze(-1/m * np.sum(Y * np.log(AL)))

    # -Y / A
    def cost_derivative(self, AL, Y):
        return AL - Y


# mean squared error - for regression
class MSE:

    def forward_cost(self, AL, Y):
        m = Y.shape[1]
        return np.squeeze(1/m * np.sum(np.square((Y - AL))))
        
    def cost_derivative(self, AL, Y):
        return -2 * (Y - AL)