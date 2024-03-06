# deepteddy - A mini deep learning framework built to teach myself about ML and Python

Much of the code is heavily inspired by a similar project by Kayden Kehe, material from Andrew Ng's deep learning specialization on Coursera.com, and the widely used Keras ML API. I have included many of my own modifications and additions, as well. It is written entirely in Python 3 using only NumPy and Python standard libraries.

To install the deepteddy framework, run the following in a command line:

``` 
pip install -i https://test.pypi.org/simple/ deepteddy 
```

Make sure you have Python version 3.8 or newer installed on your machine.

The deepteddy framework allows for easy construction of deep neural networks with many of the most widely used optimizers, initializers, regularizers, cost functions, and activation functions. To create neural networks with deepteddy, users simply have to create a Network object, add layers to the network, configure certain hyperparameters, train the network, and (optionally) save the weights and biases of the trained network to a json file for later use. You can get an idea of how the framework can be used by looking at examples in the demos folder. For example, the following shows how to set up a two-layer neural network to recognize MNIST digits, which achieved an accuracy of 99.6% on the training set and 98.4% on the test set:

```python
import numpy as np
from keras.datasets import mnist
from deepteddy import activations, costs, layers, network, optimizers, utils
from importlib import reload
from PIL import Image

reload(network)
np.random.seed(1)

# Load data
(train_x, train_y), (test_x, test_y) = mnist.load_data()

# Flatten, normalize, and reshape data for model
train_x = np.asarray(train_x.reshape(train_x.shape[0], -1) / 255).T
test_x = np.asarray(test_x.reshape(test_x.shape[0], -1) / 255).T
train_y = utils.one_hot(train_y, num_classes=10).T
test_y = utils.one_hot(test_y, num_classes=10).T

# create NN
digits = network.Network()
digits.add_layer(layers.Dense(num_nodes=200, activation=activations.ReLU()))
digits.add_layer(layers.Dense(num_nodes=train_y.shape[0], activation=activations.Sigmoid()))

# configure
digits.configure_network(
    input_layer_size=train_x.shape[0],
    cost_func=costs.CategoricalCrossEntropy(),
    optimizer=optimizers.Adam(),
)

# train
digits.train(train_x, train_y, learning_rate=0.001, epochs=24, minibatch_size=16, verbose=True)

# save parameters to json file
digits.write_parameters(name='mnist_demo_params.json', dir='')

# summary
digits.network_summary()

# accuracy
pred_train = digits.predict(train_x)
print('\nTraining Accuracy:', utils.evaluate_accuracy(pred_train, train_y))
pred_test = digits.predict(test_x)
print('Testing Accuracy:', utils.evaluate_accuracy(pred_test, test_y))

# try a picture of a digit from the digit_pics folder
new_number = Image.open('digit_pics/eight.jpeg').convert('L')
new_number = (np.array(new_number).reshape(1, -1) / 255).T
pred_num = digits.predict(new_number)
print('Handwritten Image Prediction: ', utils.argmax(pred_num))
```