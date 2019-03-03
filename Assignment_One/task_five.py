import numpy as np


def sigmoid(x, derv=False):
    if derv:
        return x * (1 - x)
    else:
        return 1 / (1 + np.exp(-x))


def relu(x):
    return max(0, x)


def tanh(x):
    return np.tanh(x)


def break_weights(weights):
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


def foreward_prop(input_value, weights_input, bias_node_hidden, weights_hidden, bias_node_output):
    # Now first layer

    input_layer_output = weights_input.dot(input_value) + bias_node_hidden
    # Activation if there is one
    input_layer_output = sigmoid(input_layer_output)

    hidden_layer_output = weights_hidden.dot(input_layer_output) + bias_node_output
    hidden_layer_output = sigmoid(hidden_layer_output)

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

    hidden_layer_output = foreward_prop(input_value, weights_input, bias_node_hidden, weights_hidden, bias_node_output)

    # Now hidden_layer_output should be one or the other

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
    Gradient Descent
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

    bias_node_hidden, bias_node_output, weights_input, weights_hidden = break_weights(weights)

    # Partial Derivatives for
    d_weights_input = 0.0
    d_weights_hidden = 0.0
    d_bias_node_hidden = 0.0
    d_bias_node_output = 0.0

    for index, input_value in enumerate(X):
        x1 = input_value[0]
        x2 = input_value[1]
        input_value = np.reshape(np.asarray([x1, x2]), (2, 1))  # Get it as a column vector

        z1 = weights_input.dot(input_value) + bias_node_hidden  # 2x2 * 2x1 + 2x1 = 2x1
        a1 = sigmoid(z1)  # 2x1

        z2 = weights_hidden.dot(a1) + bias_node_output  # 1x2 * 2x1 + 1x1 = 1x1
        a2 = sigmoid(z2)  # 1x1

        # Back prop.
        dz2 = a2 - y[index]  # 1x1
        d_weights_hidden += dz2 * a1.T  # 1x1 .* 1x2 = 1x2

        dz1 = np.multiply((weights_hidden.T * dz2), sigmoid(a1, derv=True))  # (2x1 * 1x1) .* 2x1 = 2x1
        d_weights_input += dz1.dot(input_value.T)  # 2x1 * 1x2 = 2x2

        d_bias_node_hidden += dz1  # 2x1
        d_bias_node_output += dz2  # 1x1

    d_weights_input /= 4.
    d_weights_hidden /= 4.
    d_bias_node_hidden /= 4.
    d_bias_node_output /= 4.

    gradient_weights = remake_weights(d_bias_node_hidden, d_bias_node_output, d_weights_input, d_weights_hidden)
    return gradient_weights


def grdmse_other(weights):
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
        changed_weight = weights[index] + eta
        weights[index] = changed_weight
        changed_mse, _ = mse(weights)
        gradient = (changed_mse - base_mse) / eta
        grad_weights[index] = gradient
        weights[index] -= eta

    return grad_weights


X = np.array([
    [0, 1],
    [1, 0],
    [1, 1],
    [0, 0]
])

y = np.array([
    [1],
    [1],
    [0],
    [0]
])


def train_network(size, data, labels, iterations=5000, learning_rate=0.01):
    """
    Actual gradient descent, initialize to random, then iterate over weights = weights - eta* grdmse(weights)
    :param size: Size of weights, so 9 for the XOR network, 256 for MNIST
    :param eta:
    :return:
    """

    weights = np.random.rand(size)

    # Just need MSE and grdmse

    mserror = np.zeros((iterations,1))
    misclassified = np.zeros((iterations,1))
    for i in range(iterations):
        # Get MSE with current weights
        mserror[i], misclassified[i] = mse(weights)

        # Get gradient
        gradient_weights = grdmse_other(weights)

        # update weights with gradient descent
        weights = weights - learning_rate * gradient_weights


    return weights
