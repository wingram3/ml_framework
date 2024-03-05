from importlib import reload
import numpy as np
from deepteddy import activations, costs, initializers, layers, network, optimizers, gradient_check

reload(costs)

reload(costs)

# Create dummy data
X = np.random.randn(3, 100)
Y = np.random.randint(0, 2, (1, 100))

# create network
cost = network.Network()

# add layers
cost.add_layer(layers.Dense(num_nodes=4, activation=activations.ReLU()))
cost.add_layer(layers.Dense(num_nodes=2, activation=activations.ReLU()))
cost.add_layer(layers.Dense(num_nodes=1, activation=activations.Sigmoid()))

# configure for different cost functions
# Binary Cross-Entropy
cost.configure_network(input_layer_size=3, cost_func=costs.BinaryCrossEntropy(), optimizer=optimizers.SGD())
diff = gradient_check.gradient_check(cost, X, Y)
print(diff)

# Categorical Cross Entropy
cost.configure_network(input_layer_size=3, cost_func=costs.CategoricalCrossEntropy(), optimizer=optimizers.SGD())
diff = gradient_check.gradient_check(cost, X, Y)
print(diff)

# Mean Squared Error
cost.configure_network(input_layer_size=3, cost_func=costs.MSE(), optimizer=optimizers.SGD())
diff = gradient_check.gradient_check(cost, X, Y)
print(diff)



