import numpy as np


def zeros(layer_size, prev_layer_size):
    """ Initalize parameters as all zeros. """
    return {
        'W' : np.zeros((layer_size, prev_layer_size)),
        'b' : np.zeros((layer_size, 1))
    }


def ones(layer_size, prev_layer_size):
    """ Initialize parameters to all ones. """
    return {
        'W' : np.ones((layer_size, prev_layer_size)),
        'b' : np.zeros((layer_size, 1))
    }


def random_normal(layer_size, prev_layer_size):
    """ Initilialize parameters according to a random normal distribution. """
    return {
        'W' : np.random.randn(layer_size, prev_layer_size),
        'b' : np.zeros((layer_size, 1))
    }


def random_uniform(layer_size, prev_layer_size):
    """ Initialize parameters according to a random uniform distribution. """
    return {
        'W' : np.random.uniform(layer_size, prev_layer_size),
        'b' : np.zeros((layer_size, 1))
    }


def he_normal(layer_size, prev_layer_size):
    """ Initialize parameters according to a He normal distrubution, better for ReLU activation. """
    params = {
        'W' : np.random.randn(layer_size, prev_layer_size) * np.sqrt(2 / prev_layer_size),
        'b' : np.zeros((layer_size, 1))
    }
    assert(params['W'].shape == (layer_size, prev_layer_size))
    assert(params['b'].shape == (layer_size, 1))
    return params


def he_uniform(layer_size, prev_layer_size):
    """ Initialize parameters according to a He uniform distrubution, better for ReLU activation. """
    return {
        'W' : np.random.uniform(layer_size, prev_layer_size) * np.sqrt(2. / prev_layer_size),
        'b' : np.zeros((layer_size, 1))
    }


def xavier_normal(layer_size, prev_layer_size):
    """ Initialize parameters according to a Xavier normal distrubution, better for tanh activation. """
    return {
        'W' : np.random.randn(layer_size, prev_layer_size) * np.sqrt(1. / prev_layer_size),
        'b' : np.zeros((layer_size, 1))
    }


def xavier_uniform(layer_size, prev_layer_size):
    """ Initialize parameters according to a Xavier uniform distrubution, better for tanh activation. """
    return {
        'W' : np.random.uniform(layer_size, prev_layer_size) * np.sqrt(1. / prev_layer_size),
        'b' : np.zeros((layer_size, 1))
    }


def glorot_normal(layer_size, prev_layer_size):
    """ Initilalize parameters according to the Bengio / Glorot normal distrubution, better for tanh function. """
    return {
        'W' : np.random.randn(layer_size, prev_layer_size) * np.sqrt(2. / prev_layer_size + layer_size),
        'b' : np.zeros(layer_size, 1)
    }


def glorot_uniform(layer_size, prev_layer_size):
    """ Initilalize parameters according to the Bengio / Glorot uniform distrubution, better for tanh function. """
    limit = np.sqrt(6 / (layer_size + prev_layer_size))
    return {
        'W' : np.random.uniform(-limit, limit, (layer_size, prev_layer_size)),
        'b' : np.zeros((layer_size, 1))
    }
