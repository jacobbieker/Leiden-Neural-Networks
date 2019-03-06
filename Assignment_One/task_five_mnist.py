import numpy as np
from .mnist import mnist_data

def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def relu(x):
    return max(0, x)


def tanh(x):
    return np.tanh(x)


def break_weights(weights):
    """
    Weights are for 256 input nodes, 30 hidden nodes, 10 output nodes

    So

    256 weights + 256 bias for first node

    256 * 30 weights for first node to hidden

    30 * 10 for weights from hidden to output



    30 weights and 30 bias for hidden nodes


    :param weights:
    :return:
    """
    bias_node_hidden = weights[4:6]
    bias_node_hidden = np.reshape(bias_node_hidden, (2, 1))
    bias_node_output = np.reshape(weights[8], (1, 1))

    weights_input = weights[0:4]

    weights_input = np.reshape(np.asarray(weights_input), (2, 2))
    weights_hidden = np.reshape(weights[6:8], (1, 2))
    return bias_node_hidden, bias_node_output, weights_input, weights_hidden


def remake_weights(bias_node_hidden, bias_node_output, weights_input, weights_hidden):
    weights_hidden = np.reshape(weights_hidden, (2,))
    weights_input = np.reshape(weights_input, (4,))
    bias_node_output = np.reshape(bias_node_output, (1,))
    bias_node_hidden = np.reshape(bias_node_hidden, (2,))

    weights = np.append(weights_input, [bias_node_hidden, weights_hidden, bias_node_output])

    return weights


def foreward_prop(input_value, weights_input, bias_node_hidden, weights_hidden, bias_node_output, activation):
    # Now first layer

    input_layer_output = weights_input.dot(input_value) + bias_node_hidden
    # Activation if there is one
    input_layer_output = activation(input_layer_output)

    hidden_layer_output = weights_hidden.dot(input_layer_output) + bias_node_output
    # Activation if there is one
    hidden_layer_output = activation(hidden_layer_output)

    return hidden_layer_output


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

    bias_node_hidden, bias_node_output, weights_input, weights_hidden = break_weights(weights)

    # Now take the input, multiple by the weights for each layer to get the hidden layer units

    input_value = np.reshape(np.asarray([x1, x2]), (2, 1))  # Get it as a column vector

    hidden_layer_output = foreward_prop(input_value, weights_input, bias_node_hidden, weights_hidden, bias_node_output,
                                        sigmoid)

    # Now hidden_layer_output outputs to the single output node

    return hidden_layer_output


def mse(weights):
    """
    (0,0),(0,1),(1,0),(1,1) = 0,1,1,0

    C = 1/2n * sum(y_true - weight)^2
    Mean Squared Error
    :param weights:
    :return:
    """

    X = np.array([
        [0, 0],
        [1, 0],
        [0, 1],
        [1, 1]
    ])

    y = np.array([
        [0],
        [1],
        [1],
        [0]
    ])

    mean_squared_error = 0.0

    misclassified_inputs = 0

    for index, input_value in enumerate(X):
        x1 = input_value[0]
        x2 = input_value[1]
        y_true = y[index]
        output = xor_net(x1, x2, weights)
        if output > 0.5:
            if y_true == 0:
                misclassified_inputs += 1
        elif output < 0.5:
            if y_true == 1:
                misclassified_inputs += 1
        mean_squared_error += (y_true - output) ** 2

    mean_squared_error /= 4.  # Get the mean value of the error

    return mean_squared_error, misclassified_inputs


def grdmse(weights):
    """
    Do it the change eta one
    :param weights:
    :return:
    """
    eta = 1e-3

    # Now since it is a function of 9 variables, need to do the gradient over 9 time
    # (f(1+e,2,3,4,5,6,7,8,9) - f(1,2,3,4,5,6,7,8,9))/e for example
    # Need to get mse from each for that

    base_mse, _ = mse(weights)

    grad_weights = np.zeros(weights.shape)
    for index, weight in enumerate(weights):
        weights[index] = weights[index] + eta
        changed_mse, _ = mse(weights)
        gradient = (changed_mse - base_mse) / eta
        grad_weights[index] = gradient
        # Go back to default value
        weights[index] -= eta

    return grad_weights


def train_network(size, iterations=5000, learning_rate=0.01):
    """
    Actual gradient descent, initialize to random, then iterate over weights = weights - eta* grdmse(weights)
    :param size: Size of weights, so 9 for the XOR network, 256 for MNIST
    :param eta:
    :return:
    """

    weights = np.random.rand(size)

    num_inputs = 256
    num_hidden = 30
    num_output = 10

    weights_one = np.random.normal(0, 1, (num_hidden, num_inputs))
    weights_two = np.random.normal(0, 1, (num_output, num_hidden))
    bias_one = np.random.random((num_hidden, 1))
    bias_two = np.random.random((num_output, 1))

    # Just need MSE and grdmse

    mserror = np.zeros((iterations, 1))
    misclassified = np.zeros((iterations, 1))
    for i in range(iterations):
        # Get MSE with current weights
        mserror[i], misclassified[i] = mse(weights)

        # Get gradient
        gradient_weights = grdmse(weights)

        # update weights with gradient descent
        weights = weights - learning_rate * gradient_weights



    return weights
