import numpy as np
from mnist import mnist_data
from .perceptron import Perceptron


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def relu(x):
    return max(0, x)


def tanh(x):
    return np.tanh(x)


def xor_net(x1, x2, weights):
    """
    Two input nodes, two hidden nodes, and one output node
    :param x1:
    :param x2:
    :param weights:
    :return:
    """

    return NotImplementedError


def mse(weights, y_true):
    """
    (0,0),(0,1),(1,0),(1,1) = 0,1,1,0

    C = 1/2n * sum(y_true - weight)^2
    Mean Squared Error
    :param weights:
    :return:
    """

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
