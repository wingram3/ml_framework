import numpy as np
from deepteddy import activations, costs, layers, network, optimizers, regularizers, utils
from importlib import reload
from h5py import File

reload(network)
np.random.seed(0)

# load cat data
train_dataset = File('datasets/train_catvnoncat.h5', 'r')
train_x = np.array(train_dataset['train_set_x'][:]) # Train set features
train_y = np.array(train_dataset['train_set_y'][:]) # Train set labels

test_dataset = File('datasets/test_catvnoncat.h5', 'r')
test_x = np.array(test_dataset['test_set_x'][:]) # Test set features
test_y = np.array(test_dataset['test_set_y'][:]) # Test set labels

train_y = train_y.reshape((train_y.shape[0], 1)).T
test_y = test_y.reshape((test_y.shape[0], 1)).T

# Flatten and normalize
train_x = (train_x.reshape(train_x.shape[0], -1) / 255).T
test_x = (test_x.reshape(test_x.shape[0], -1) / 255).T

# check dims
print(train_x.shape)
print(train_y.shape)
print(test_x.shape)
print(test_y.shape)

# create NN
digits = network.Network()
digits.add_layer(layers.Dense(num_nodes=20, activation=activations.ReLU(), regularizer=regularizers.L2(lmbda=0.1)))
digits.add_layer(layers.Dense(num_nodes=7, activation=activations.ReLU(), regularizer=regularizers.L2(lmbda=0.1)))
digits.add_layer(layers.Dense(num_nodes=5, activation=activations.ReLU(), regularizer=regularizers.L2(lmbda=0.1)))
digits.add_layer(layers.Dense(num_nodes=1, activation=activations.Sigmoid(), regularizer=regularizers.L2(lmbda=0.1)))

# configure
digits.configure_network(
    input_layer_size=train_x.shape[0],
    cost_func=costs.MSE(),
    optimizer=optimizers.Adam(),
)

# train
digits.train(train_x, train_y, learning_rate=0.001, epochs=500, minibatch_size=None, verbose=True)

# save parameters to json file
# digits.write_parameters(name='cats_demo_params.json', dir='')

# summary
digits.network_summary()

# accuracy
pred_train = digits.predict(train_x)
print('\nTraining Accuracy:', utils.evaluate_accuracy(pred_train, train_y))
pred_test = digits.predict(test_x)
print('Testing Accuracy:', utils.evaluate_accuracy(pred_test, test_y))