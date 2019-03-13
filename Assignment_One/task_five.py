import numpy as np
import matplotlib.pyplot as plt

np.random.seed(1)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def relu(x):
    return np.maximum(np.zeros(x.shape), x)


def tanh(x):
    return np.tanh(x)

x_values = np.linspace(-10,10)
relu_vals = relu(x_values)
tanh_vals = tanh(x_values)
sigmoid_vals = sigmoid(x_values)

plt.plot(x_values, relu_vals, label="ReLU")
plt.plot(x_values, tanh_vals, label="Tanh")
plt.plot(x_values, sigmoid_vals, label="Sigmoid")
plt.xlabel("Input")
plt.ylabel("Output")
plt.legend(loc='best')
plt.title("Activation Functions")
#plt.tight_layout()
plt.ylim((-1.,1.5))
plt.savefig("Activations.png", dpi=300)
plt.show()

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
    Gradient Descent by changing eta

    :param weights:
    :return:
    """
    eta = 1e-5

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


def train_network(size, iterations=100000, learning_rate=0.1, init_low=-1.5, init_high=1.5, init_method=np.random.uniform):
    """
    Trains the XOR network
    :param size: Size of the weights, 9 for XOR, 256 for MNIST (not implemented)
    :param iterations: Iterations to run for
    :param learning_rate: Learning rate
    :param init_low: Low value for the initialization distribution
    :param init_high: High value for the initialization distribution
    :param init_method: Method of initialization
    :return:
    """

    weights = init_method(low=init_low, high=init_high, size=size)

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

    iterations = [i for i in range(iterations)]

    return weights, iterations, mserror, misclassified


learning_rates = [1.,0.1,0.01,0.001]
weight_finals = []
iterations_final = []
mserror_final = []
misclassified_final = []
for lr in learning_rates:
    np.random.seed(1)
    weights, iterations, mserror, misclassified = train_network(9, learning_rate=lr)
    weight_finals.append(weights)
    iterations_final.append(iterations)
    mserror_final.append(mserror)
    misclassified_final.append(misclassified)

fig, axes = plt.subplots(2, 1, sharex="all")
for index, lr in enumerate(learning_rates):
    axes[0].plot(iterations_final[index], mserror_final[index], label="LR: {}".format(lr))
    axes[0].set_xlabel("Iteration")
    axes[0].set_ylabel("Mean Squared Error")
    axes[0].legend(loc="best", fontsize="xx-small")
    axes[1].plot(iterations_final[index], misclassified_final[index], label="LR: {}".format(lr))
    axes[1].set_xlabel("Iteration")
    axes[1].set_ylabel("Number Misclassified")
    axes[1].legend(loc="best", fontsize="xx-small")

fig.suptitle("Initializer: Uniform Act: sigmoid")
fig.savefig("Task5_sigmoid_lr.png", dpi=300)
fig.show()

weight_finals = []
iterations_final = []
mserror_final = []
misclassified_final = []
lows = [-1.5, -1., 0.0, 0.5]
highs =  [1.0, 2.0]

for low in lows:
    for high in highs:
        np.random.seed(1)
        weights, iterations, mserror, misclassified = train_network(9, init_low=low, init_high=high)
        weight_finals.append(weights)
        iterations_final.append(iterations)
        mserror_final.append(mserror)
        misclassified_final.append(misclassified)

# Now plot
fig, axes = plt.subplots(2, 1, sharex="all")
for l_index, low in enumerate(lows):
    for h_index, high in enumerate(highs):
        axes[0].plot(iterations_final[l_index*h_index], mserror_final[l_index*h_index], label="Uniform: L: {} H: {}".format(low, high))
        axes[0].set_xlabel("Iteration")
        axes[0].set_ylabel("Mean Squared Error")
        axes[0].legend(loc="best", fontsize="xx-small")
        #axes[0].set_aspect('equal', adjustable='box')
        axes[1].plot(iterations_final[l_index*h_index], misclassified_final[l_index*h_index], label="Uniform: L: {} H: {}".format(low, high))
        axes[1].set_xlabel("Iteration")
        axes[1].set_ylabel("Number Misclassified")
        axes[1].legend(loc="best", fontsize="xx-small")
        #axes[1].set_aspect('equal', adjustable='box')

fig.suptitle("Initializer: Uniform Act: sigmoid")
fig.savefig("Task5_sigmoid.png", dpi=300)
fig.show()
