import numpy as np
import deepteddy
import inspect

'''
def import_cupy():
    global np
    np = __import__('cupy')
'''

# MISCELLANEOUS UTILITY FUNCTIONS

# one-hot function
def one_hot(Y, num_classes):
    return np.eye(num_classes)[Y.reshape(-1)]
    
def binary_round(Y):
        return np.squeeze(np.where(Y > 0.5, 1, 0))
    
def argmax(Y):
    return np.squeeze(np.argmax(Y, axis=0))
    
# evaluate network accuracy
def evaluate_accuracy(Y_pred, Y):
    Y_pred = binary_round(Y_pred) if Y_pred.shape[0] == 1 else argmax(Y_pred)
    Y = binary_round(Y) if Y.shape[0] == 1 else argmax(Y)
    return np.mean(np.where(Y_pred == Y, 1, 0))

def configure_cupy():
    for module, _ in inspect.getmembers(deepteddy, inspect.ismodule):
        exec(f'sandbox.{module}.import_cupy()')