import numpy as np
from mnist import mnist_data
from .perceptron import Perceptron

def sigmoid(x):
    return 1 / (1 + np.exp(-x))


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

    error = 0.5*1 * np.sum((y_true - weights)**2)
    return error


def grdmse(weights):


    return NotImplementedError
