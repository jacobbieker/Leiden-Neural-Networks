import numpy as np
from mnist import mnist_data

class Perceptron(object):

    def __init__(self, num_inputs, epochs=500, learning_rate=0.01):
        self.weights = np.random.uniform(low=-1., high=1., size=(num_inputs+1)*10).reshape(257,10)
        self.weights[-1:,:] = 1
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.train_history = []
        self.test_history = []

    def predict(self, inputs):
        # Now the matrix mult with the weights to get the output
        output = np.dot(inputs, self.weights)

        # Now need to choose which one is the largest to give the output for this one
        activation = np.argmax(output)

        # Return the activation
        return activation

    def predict_on_set(self, test_inputs, labels, verbose=False):
        training_inputs = np.c_[test_inputs, np.ones(test_inputs.shape[0])] # Add biases
        combined = np.asarray(list(zip(training_inputs, labels)))
        right = 0
        wrong = 0
        for inputs, label in combined:
            prediction = self.predict(inputs)
            if prediction == label:
                right += 1
            else:
                wrong += 1
        if verbose:
            print((right/(wrong+right)))
        return (right/(wrong+right))


    def train(self, training_inputs, labels, shuffle=False, neg_pred=10, pos_pred=1):
        training_inputs = np.c_[training_inputs, np.ones(training_inputs.shape[0])] # Add biases
        combined = np.asarray(list(zip(training_inputs, labels)))
        best_test = 0.0
        prev_test = 0.0
        for _ in range(self.epochs):
            right = 0
            wrong = 0
            self.test_history.append(self.predict_on_set(x_test, y_test))
            if shuffle:
                np.random.shuffle(combined)
            for inputs, label in combined:
                ldir = np.zeros((10,1))
                prediction = self.predict(inputs)
                label = label[0]
                if prediction == label:
                    ldir[label] += pos_pred
                else:
                    ldir[label] += neg_pred
                    ldir[prediction] -= neg_pred

                if prediction == label:
                    right += 1
                else:
                    wrong += 1
                self.weights[:-1] += self.learning_rate * (ldir * inputs[:-1]).T
                self.weights[-1:] += self.learning_rate * ldir.T
            self.train_history.append(self.acc(right, wrong))
            #print(right/(right + wrong))
        self.plot_training(neg_pred, pos_pred)

    def plot_training(self, neg_val, pos_val):
        import matplotlib.pyplot as plt
        timesteps = [i for i in range(len(self.train_history))]
        plt.plot(timesteps, self.train_history, label='Training Set')
        plt.plot(timesteps, self.test_history, label='Test Set')
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.title("Accuracy over Epoch Neg: {} Pos: {}".format(neg_val, pos_val))
        plt.legend(loc='best')
        plt.savefig("Neg_{}_Pos_{}_Test_{}_Train_{}.png".format(neg_val, pos_val, self.test_history[-1], self.train_history[-1]), dpi=300)
        plt.cla()
        #plt.show()


    def acc(self, right, wrong):
        return right/(right + wrong)


x_train, y_train, x_test, y_test = mnist_data("data")

print(x_train.shape)
# Reshape to 256 elements
x_train = x_train.reshape((-1,256))
print(x_train.shape)
x_test = x_test.reshape((-1,256))

np.random.seed(1337)

network = Perceptron(256)

network.predict_on_set(x_test, y_test)
network.train(x_train, y_train, shuffle=True)
network.predict_on_set(x_test, y_test)

trained_precdictions = []
for neg_val in np.logspace(-5, 5, num=10):
    for pos_val in np.logspace(-5, 5, num=10):
        print("Neg Val: {} Pos Val: {}".format(neg_val, pos_val))
        np.random.seed(1337)
        network = Perceptron(256)
        network.predict_on_set(x_test, y_test, verbose=True)
        network.train(x_train, y_train, shuffle=True, neg_pred=neg_val, pos_pred=pos_val)
        trained_precdictions.append(network.predict_on_set(x_test, y_test, verbose=True))

# Now have all the predictions
print(np.max(trained_precdictions))
print(np.min(trained_precdictions))
