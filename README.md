# deepteddy - A mini deep learning framework built to teach myself about ML and Python

Much of the code is heavily inspired by a similar project by Kayden Kehe, material from Andrew Ng's deep learning specialization on Coursera.com, and the widely used Keras ML API. I have included many of my own modifications and additions, as well. It is written entirely in Python 3 using only NumPy and Python standard libraries.

To install the deepteddy framework, run the following in a command line:

``` 
pip install -i https://test.pypi.org/simple/ deepteddy 
```

Make sure you have Python version 3.8 or newer installed on your machine.

The deepteddy framework allows for easy construction of deep neural networks with many of the most widely used optimizers, initializers, regularizers, cost functions, and activation functions. To create neural networks with deepteddy, users simply have to create a Network object, add layers to the network, configure certain hyperparameters, train the network, and (optionally) save the weights and biases of the trained network to a json file for later use. You can get an idea of how the framework can be used by looking at examples in the demos folder.
