from importlib import reload
import numpy as np
from deepteddy import activations, costs, initializers, layers, network, optimizers, Wing.grad_check.gradient_check as gradient_check

reload(activations)

# Create dummy data
X = np.random.randn(3, 100)
Y = np.random.randint(0, 2, (1, 100))


# create network
activation = network.Network()
activation.add_layer(layers.Dense(num_nodes=10, activation=activations.ReLU()))
activation.add_layer(layers.Dense(num_nodes=10, activation=activations.Softmax()))
activation.add_layer(layers.Dense(num_nodes=1, activation=activations.Sigmoid()))

# configure
activation.configure_network(input_layer_size=3, cost_func=costs.BinaryCrossEntropy(), optimizer=optimizers.SGD())

# check gradient for different activations
diff = gradient_check.gradient_check(activation, X, Y)
print(diff)