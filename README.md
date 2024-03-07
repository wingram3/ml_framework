# deepteddy - A mini deep learning framework built to teach myself about ML and Python

The deepteddy framework allows for easy construction of deep neural networks with many of the most widely used optimizers, initializers, regularizers, cost functions, and activation functions. To create neural networks with deepteddy, users simply have to create a Network object, add layers to the network, configure certain hyperparameters, train the network, and (optionally) save the weights and biases of the trained network to a json file for later use. You can get an idea of how the framework can be used by looking at examples in the demos folder.

This project is written entirely in Python 3 using only NumPy and Python standard libraries.

To install the deepteddy framework, run the following in a command line:

``` 
pip install -i https://test.pypi.org/simple/ deepteddy==0.0.7
```

Once installed, import the package in a Python file as follows:

```
import deepteddy
```

Make sure you have Python version 3.8 or newer installed on your machine.