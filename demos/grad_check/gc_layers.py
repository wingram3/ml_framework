from importlib import reload
import numpy as np
from deepteddy import activations, costs, initializers, layers, network, optimizers, Wing.grad_check.gradient_check as gradient_check

# Create dummy data
X = np.random.randn(3, 100)
Y = np.random.randint(0, 2, (1, 100))

# create network
layer = network.Network()

# add layers
layer.add_layer(layers.Dense(num_nodes=4, activation=activations.ReLU()))
layer.add_layer(layers.Dense(num_nodes=2, activation=activations.ReLU()))
layer.add_layer(layers.Dense(num_nodes=1, activation=activations.Sigmoid()))

# configure
layer.configure_network(input_layer_size=3, cost_func=costs.BinaryCrossEntropy(), optimizer=optimizers.SGD())

# gradient check
diff = gradient_check.gradient_check(layer, X, Y)
print(diff)