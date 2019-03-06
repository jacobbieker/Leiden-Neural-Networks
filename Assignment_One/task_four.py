import numpy as np
from mnist import mnist_data

class Perceptron(object):

    def __init__(self, num_inputs, epochs=5000, learning_rate=0.001):
        self.weights = np.random.uniform(low=-1., high=1., size=(num_inputs+1)*10).reshape(257,10)
        self.weights[-1:,:] = 1
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
        num_before_update = 10
        combined = np.asarray(list(zip(training_inputs, labels)))
        for _ in range(self.epochs):
            right = 0
            wrong = 0
            np.random.shuffle(combined)
            num_batch = int(np.floor(len(combined)/num_before_update))

            for batch in range(num_batch):
                large_input = np.zeros(257)
                for inputs, label in combined[batch*num_before_update:(batch+1)*num_before_update]:
                    ldir = np.zeros((10,1))
                    prediction = self.predict(inputs)
                    label = label[0]
                    np.add(large_input, inputs)
                    if prediction == label:
                        ldir[label] += 1
                    else:
                        ldir[label] += 1
                        ldir[prediction] -= 1
                    #print(ldir)

                # TODO Fix something here
                    if prediction == label:
                        right += 1
                    else:
                        wrong += 1
                # Add shuffling, get 100 random samples each time, add +1 or -1 each time for each one to ldir, then update
                # at once, just adding +1, -1 worked to get up and changing a bit more
                # Update the weights
                #print(self.weights[-1:,:].shape)
                #exit()
                    self.weights[:-1] += self.learning_rate * (ldir * inputs[:-1]).T
                    self.weights[-1:] += self.learning_rate * ldir.T
                #training_inputs[:,-1] += self.learning_rate * (prediction - label)
            print(right/(right + wrong))

x_train, y_train, x_test, y_test = mnist_data("data")

print(x_train.shape)
# Reshape to 256 elements
x_train = x_train.reshape((-1,256))
print(x_train.shape)
x_test = x_test.reshape((-1,256))

network = Perceptron(256)

network.train(x_train, y_train)