from deepteddy import network
import inspect
import numpy as np

# Gradient checking
def gradient_check(network, X, Y, epsilon=1e-4):

    # Calculate actual gradient
    AL = network.forward_propagate(X)
    grad = network.backpropagate(AL, Y)

    # Declare empty arrays for gradient and gradient approximation
    num_params = network.network_summary(print_table=False)
    grad_arr, grad_aprox_arr = np.zeros(num_params), np.zeros(num_params)

    # Loop through every parameter
    iter = 0
    for layer in range(len(network.layers)):
        for param_type in ['W', 'b']:
            for param in range(network.parameters[layer][param_type].shape[0]):

                # Calculate the cost with the parameter shifted by +epsilon
                network.parameters[layer][param_type][param][0] += epsilon
                AL_pe = network.forward_propagate(X)
                cost_pe = network.cost_func.forward_cost(AL_pe, Y)

                # Calculate the cost with the parameter shifted by -epsilon
                network.parameters[layer][param_type][param][0] -= 2 * epsilon
                AL_ne = network.forward_propagate(X)
                cost_ne = network.cost_func.forward_cost(AL_ne, Y)

                # Reset the parameter
                network.parameters[layer][param_type][param][0] += epsilon

                # Calculate the approximate parameter derivative w.r.t. cost
                grad_aprox_arr[iter] = (cost_pe - cost_ne) / (2 * epsilon)

                # Append actual gradient value to list (allows for Euclidean distance in later step)
                grad_arr[iter] = grad[layer]['d' + param_type][param][0]

                iter += 1

            # Return representation of distance between actual and approximate gradients
            return np.linalg.norm(grad_arr - grad_aprox_arr) / (np.linalg.norm(grad_arr) + np.linalg.norm(grad_aprox_arr))