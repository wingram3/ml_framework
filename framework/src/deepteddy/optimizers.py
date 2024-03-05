import numpy as np

'''
def import_cupy():
    global np
    np = __import__('cupy')
'''
    
# stochastic gradient descent
class SGD:

    def __init__(self, learning_rate=0.001):
        self.learning_rate = learning_rate

    def build(self, parameters, layers):
        return parameters

    def update_step(self, parameters, layers, gradients, X):
        for layer in range(len(layers)):
            m = X.shape[1]
            parameters[layer]['W'] -= self.learning_rate * gradients[layer]['dW']
            parameters[layer]['b'] -= self.learning_rate * gradients[layer]['db']  
            
        return parameters


# root mean squared propagation - essentially penalizes the learning rate as the cost function nears local minima
class RMSProp:

    def __init__(self, learning_rate=0.001, beta=0.9, epsilon=1e-8):
        self.learning_rate = learning_rate
        self.beta = beta
        self.epsilon = epsilon

    # initialize learning rate scalers
    def build(self, parameters, layers):
        for layer in range(len(layers)):
            parameters[layer]['SdW'] = np.zeros(parameters[layer]['W'].shape)
            parameters[layer]['Sdb'] = np.zeros(parameters[layer]['b'].shape)

        return parameters
    
    # update parameters, dividing the learning rate by the square root of the scaler, plus epsilon to prevent divide by zero errors
    def update_step(self, parameters, layers, gradients, X):
        for layer in range(len(layers)):
            parameters[layer]['SdW'] = self.beta*parameters[layer]['SdW'] * (1 - self.beta)*np.square(gradients[layer]['dW'])
            parameters[layer]['Sdb'] = self.beta*parameters[layer]['Sdb'] * (1 - self.beta)*np.square(gradients[layer]['db'])

            parameters[layer]['W'] -= self.learning_rate * parameters[layer]['SdW'] / (np.sqrt(parameters[layer]['SdW']) + self.epsilon)
            parameters[layer]['b'] -= self.learning_rate * parameters[layer]['Sdb'] / (np.sqrt(parameters[layer]['Sdb']) + self.epsilon)

        return parameters


# adaptive moment estimation - combines momentum with RMSProp
class Adam:

    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.learning_rate = learning_rate 
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon

    def build(self, parameters, layers):
        for layer in range(len(layers)):
            parameters[layer]['VdW'] = np.zeros(parameters[layer]['W'].shape)  # initiliaze weight velocities
            parameters[layer]['Vdb'] = np.zeros(parameters[layer]['b'].shape)  # initialize bias velocities
            parameters[layer]['SdW'] = np.zeros(parameters[layer]['W'].shape)  # initialize leanring scaler for weights
            parameters[layer]['Sdb'] = np.zeros(parameters[layer]['b'].shape)  # initialize learning scaler for biases

        return parameters
    
    def update_step(self, parameters, layers, gradients, X):
        for layer in range(len(layers)):
            parameters[layer]['VdW'] = self.beta1*parameters[layer]['VdW'] + (1 - self.beta1)*gradients[layer]['dW']
            parameters[layer]['Vdb'] = self.beta1*parameters[layer]['Vdb'] + (1 - self.beta1)*gradients[layer]['db']
            parameters[layer]['SdW'] = self.beta2*parameters[layer]['SdW'] + (1 - self.beta2)*np.square(gradients[layer]['dW'])
            parameters[layer]['Sdb'] = self.beta2*parameters[layer]['Sdb'] + (1 - self.beta2)*np.square(gradients[layer]['db'])

            parameters[layer]['W'] -= self.learning_rate * parameters[layer]['VdW'] / (np.sqrt(parameters[layer]['SdW']) + self.epsilon)
            parameters[layer]['b'] -= self.learning_rate * parameters[layer]['Vdb'] / (np.sqrt(parameters[layer]['Sdb']) + self.epsilon)

        return parameters
