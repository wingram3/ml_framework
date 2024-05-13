import numpy as np
import deepteddy
import inspect


def one_hot(Y, num_classes):
    """ Convert a value to a one-hot vector. """
    return np.eye(num_classes)[Y.reshape(-1)]

def binary_round(Y):
    """ Round up if value is greater than 0.5. """
    return np.squeeze(np.where(Y > 0.5, 1, 0))

def argmax(Y):
    """ Return the max value from a vector. """
    return np.squeeze(np.argmax(Y, axis=0))

def evaluate_accuracy(Y_pred, Y):
    """ Evalute network accuracy. """
    Y_pred = binary_round(Y_pred) if Y_pred.shape[0] == 1 else argmax(Y_pred)
    Y = binary_round(Y) if Y.shape[0] == 1 else argmax(Y)
    return np.mean(np.where(Y_pred == Y, 1, 0))
