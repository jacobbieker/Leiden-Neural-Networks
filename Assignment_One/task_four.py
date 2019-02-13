import numpy as np
from mnist import mnist_data

class Perceptron(object):

    def __init__(self, num_inputs, epochs=100, learning_rate=0.01):
        self.weights = np.zeros(num_inputs+1)
        self.epochs = epochs
        self.learning_rate = learning_rate

    def predict(self, inputs):
        inputs_plus_bias = np.append(inputs, np.ones(inputs.shape[1]))
        # Now the matrix mult with the weights to get the output
        output = np.dot(inputs_plus_bias, self.weights)

        # Now need to choose which one is the largest to give the output for this one
        activation = np.argmax(output)

        # Return the activation
        return activation

    def train(self, training_inputs, labels):
        for _ in range(self.epochs):
            for inputs, label in zip(training_inputs, labels):
                prediction = self.predict(inputs)
                self.weights[1:] += self.learning_rate * (label - prediction) + inputs[:-1]
                self.weights[0] += self.learning_rate * (label - prediction)
