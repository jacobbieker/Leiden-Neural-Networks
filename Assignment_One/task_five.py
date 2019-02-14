import numpy as np
from mnist import mnist_data


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def xor_net(x1, x2, weights):
    return NotImplementedError


def mse(weights):
    return NotImplementedError


def grdmse(weights):
    return NotImplementedError
