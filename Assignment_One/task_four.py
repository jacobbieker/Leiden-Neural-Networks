import numpy as np
from mnist import mnist_data

class Perceptron(object):

    def __init__(self, num_inputs, epochs=500, learning_rate=0.001):
        self.weights = np.random.uniform(low=-1.0, high=1.0, size=(num_inputs+1)*10).reshape(257,10)
        self.epochs = epochs
        self.learning_rate = learning_rate

    def predict(self, inputs):
        # Now the matrix mult with the weights to get the output
        output = np.dot(inputs, self.weights)

        # Now need to choose which one is the largest to give the output for this one
        activation = np.argmax(output)

        # Return the activation
        return activation

    def train(self, training_inputs, labels):
        print(training_inputs.shape)
        training_inputs = np.c_[training_inputs, np.ones(training_inputs.shape[0])] # Add biases
        print(training_inputs.shape)
        for _ in range(self.epochs):
            right = 0
            wrong = 0
            for inputs, label in zip(training_inputs, labels):
                prediction = self.predict(inputs)
                label = label[0]
                # TODO Fix something here
                if prediction == label:
                    right += 1
                else:
                    wrong += 1

                # Update the weights
                self.weights[:-1, label] += self.learning_rate * (prediction - label) * inputs[:-1]
                self.weights[-1:, label] += self.learning_rate * (prediction - label)
                #training_inputs[:,-1] += self.learning_rate * (prediction - label)
            print(right/wrong)

x_train, y_train, x_test, y_test = mnist_data("data")

print(x_train.shape)
# Reshape to 256 elements
x_train = x_train.reshape((-1,256))
print(x_train.shape)
x_test = x_test.reshape((-1,256))

network = Perceptron(256)

network.train(x_train, y_train)