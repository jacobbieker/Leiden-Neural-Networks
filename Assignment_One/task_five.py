import numpy as np
from mnist import mnist_data
from .perceptron import Perceptron


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def relu(x):
    return max(0, x)


def tanh(x):
    return np.tanh(x)

def remake_weights(weights):
    bias_node_hidden = weights[4:6]
    bias_node_hidden = np.reshape(bias_node_hidden, (2,1))
    bias_node_output = np.reshape(weights[8],(1,1))

    weights_input = weights[0:4]

    weights_input = np.reshape(np.asarray(weights_input), (2,2))
    weights_hidden = np.reshape(weights[6:8], (1,2))
    return bias_node_hidden, bias_node_output, weights_input, weights_hidden


def xor_net(x1, x2, weights):
    """
    Two input nodes, two hidden nodes, and one output node

    Each non-input node takes one of the weights, one from a bias node
    one from each input node
    And one to the output node, so each hidden node has 4 connections (4 of the weights)
    Output node has 3 (2 to the hidden nodes, 1 to the bias)
    = 9 weights total
    :param x1: Either 1 or 0
    :param x2: Either 1 or 0
    :param weights: 9 values
    :return:
    """

    # Split the weights into the bias and weight arrays

    bias_node_hidden, bias_node_output, weights_input, weights_hidden = remake_weights(weights)


    # Now take the input, multiple by the weights for each layer to get the hidden layer units

    input_value = np.reshape(np.asarray([x1,x2]), (2,1)) # Get it as a column vector

    # Now first layer

    input_layer_output = weights_input.dot(input_value) + bias_node_hidden
    # Activation if there is one
    input_layer_output = sigmoid(input_layer_output)

    hidden_layer_output = weights_hidden.dot(input_layer_output) + bias_node_output
    hidden_layer_output = sigmoid(hidden_layer_output)

    # Now hidden_layer_output should be one or the other

    if hidden_layer_output > 0.5:
        return True
    else:
        return False

def mse(weights, y_true):
    """
    (0,0),(0,1),(1,0),(1,1) = 0,1,1,0

    C = 1/2n * sum(y_true - weight)^2
    Mean Squared Error
    :param weights:
    :return:
    """

    bias_node_hidden, bias_node_output, weights_input, weights_hidden = remake_weights(weights)

    error = 0.5 * 1 * np.sum((y_true - weights) ** 2)
    return error


def grdmse(weights):
    """
    Gradient Descent
    :param weights:
    :return:
    """

    e = 1E-3


    return NotImplementedError


def train_network(size, data, labels):
    """
    Actual gradient descent, initialize to random, then iterate over weights = weights - eta* grdmse(weights)
    :param size: Size of weights, so 9 for the XOR network, 256 for MNIST
    :param eta:
    :return:
    """

    weights = np.random.rand(size)




    return NotImplementedError
