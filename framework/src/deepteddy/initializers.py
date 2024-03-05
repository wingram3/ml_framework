import numpy as np

'''
def import_cupy():
    global np
    np = __import__('cupy')
'''

# initalize parameters as all zeros
def zeros(layer_size, prev_layer_size):
    return {
        'W' : np.zeros((layer_size, prev_layer_size)),
        'b' : np.zeros((layer_size, 1))
    }


# initialize parameters as all ones
def ones(layer_size, prev_layer_size):
    return {
        'W' : np.ones((layer_size, prev_layer_size)),
        'b' : np.zeros((layer_size, 1))
    }


# initilialize parameters according to a random normal distribution
def random_normal(layer_size, prev_layer_size):
    return {
        'W' : np.random.randn(layer_size, prev_layer_size),
        'b' : np.zeros((layer_size, 1))
    }


# initialize parameters according to a random uniform distribution
def random_uniform(layer_size, prev_layer_size):
    return {
        'W' : np.random.uniform(layer_size, prev_layer_size),
        'b' : np.zeros((layer_size, 1))
    }


# initialize parameters according to a He normal distrubution, better for ReLU activation
def he_normal(layer_size, prev_layer_size):
    params = {
        'W' : np.random.randn(layer_size, prev_layer_size) * np.sqrt(2 / prev_layer_size),
        'b' : np.zeros((layer_size, 1))
    }

    assert(params['W'].shape == (layer_size, prev_layer_size))
    assert(params['b'].shape == (layer_size, 1))

    return params


# initialize parameters according to a He uniform distrubution, better for ReLU activation
def he_uniform(layer_size, prev_layer_size):
    return {
        'W' : np.random.uniform(layer_size, prev_layer_size) * np.sqrt(2. / prev_layer_size),
        'b' : np.zeros((layer_size, 1))
    }


# initialize parameters according to a Xavier normal distrubution, better for tanh activation
def xavier_normal(layer_size, prev_layer_size):
    return {
        'W' : np.random.randn(layer_size, prev_layer_size) * np.sqrt(1. / prev_layer_size),
        'b' : np.zeros((layer_size, 1))
    }


# initialize parameters according to a Xavier uniform distrubution, better for tanh activation
def xavier_uniform(layer_size, prev_layer_size):
    return {
        'W' : np.random.uniform(layer_size, prev_layer_size) * np.sqrt(1. / prev_layer_size),
        'b' : np.zeros((layer_size, 1))
    }


# initilalize parameters according to the Bengio / Glorot normal distrubution, better for tanh function
def glorot_normal(layer_size, prev_layer_size):
    return {
        'W' : np.random.randn(layer_size, prev_layer_size) * np.sqrt(2. / prev_layer_size + layer_size),
        'b' : np.zeros(layer_size, 1)
    }

# initilalize parameters according to the Bengio / Glorot uniform distrubution, better for tanh function
def glorot_uniform(layer_size, prev_layer_size):
    limit = np.sqrt(6 / (layer_size + prev_layer_size))
    return {
        'W' : np.random.uniform(-limit, limit, (layer_size, prev_layer_size)),
        'b' : np.zeros((layer_size, 1))
    }