import numpy as np
import matplotlib.pyplot as plt
from mnist import mnist_data


class Perceptron(object):

    def __init__(self, num_inputs, epochs=500, learning_rate=0.1):
        self.weights = np.random.uniform(low=-1., high=1., size=(num_inputs + 1) * 10).reshape(257, 10)
        self.weights[-1:, :] = 1
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
        training_inputs = np.c_[test_inputs, np.ones(test_inputs.shape[0])]  # Add biases
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
            print((right / (wrong + right)))
        return (right / (wrong + right))

    def train(self, training_inputs, labels, shuffle=False, neg_pred=10, pos_pred=1):
        training_inputs = np.c_[training_inputs, np.ones(training_inputs.shape[0])]  # Add biases
        combined = np.asarray(list(zip(training_inputs, labels)))
        for _ in range(self.epochs):
            right = 0
            wrong = 0
            self.test_history.append(self.predict_on_set(x_test, y_test))
            if shuffle:
                np.random.shuffle(combined)
            for inputs, label in combined:
                ldir = np.zeros((10, 1))
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
        self.plot_training(neg_pred, pos_pred)
        return self.train_history, self.test_history

    def plot_training(self, neg_val, pos_val):
        timesteps = [i for i in range(len(self.train_history))]
        plt.plot(timesteps, self.train_history, label='Training Set')
        plt.plot(timesteps, self.test_history, label='Test Set')
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.title("Accuracy over Epoch Neg: {} Pos: {}".format(np.round(neg_val, 6), np.round(pos_val, 6)))
        plt.legend(loc='best')
        plt.savefig("Neg_{}_Pos_{}_Test_{}_Train_{}.png".format(neg_val, pos_val, self.test_history[-1],
                                                                self.train_history[-1]), dpi=300)
        plt.cla()

    def acc(self, right, wrong):
        return right / (right + wrong)


x_train, y_train, x_test, y_test = mnist_data("data")

# Reshape to 256 elements
x_train = x_train.reshape((-1, 256))
x_test = x_test.reshape((-1, 256))

np.random.seed(1337)

network = Perceptron(256)

network.predict_on_set(x_test, y_test)
network.train(x_train, y_train, shuffle=True)
network.predict_on_set(x_test, y_test)

trained_precdictions = []
test_predictions = []
for neg_val in np.logspace(-5, 5, num=10):
    for pos_val in np.logspace(-5, 5, num=10):
        print("Neg Val: {} Pos Val: {}".format(neg_val, pos_val))
        np.random.seed(1337)
        network = Perceptron(256)
        network.predict_on_set(x_test, y_test, verbose=True)
        train_hist, test_hist = network.train(x_train, y_train, shuffle=True, neg_pred=neg_val, pos_pred=pos_val)
        trained_precdictions.append(train_hist)
        test_predictions.append(test_hist)

# Now have all the predictions
# Plot them all
vals = np.logspace(-5, 5, num=10)

fig, axes = plt.subplots(10, 10, sharex="all", sharey="all", figsize=(20, 20))
fig.subplots_adjust(wspace=0)
# Now go through and plot everything
for neg_val in range(10):
    for pos_val in range(10):
        axes[neg_val, pos_val].plot([i for i in range(len(trained_precdictions[neg_val + pos_val]))],
                                    trained_precdictions[neg_val + pos_val])
        axes[neg_val, pos_val].plot([i for i in range(len(test_predictions[neg_val + pos_val]))],
                                    test_predictions[neg_val + pos_val])
fig.text(0.5, 0.005, 'Epoch, pos_pred value', ha='center', va='center')
fig.text(0.005, 0.5, 'Fraction Correct, neg_pred value', ha='center', va='center', rotation='vertical')
for index, val in enumerate(vals):
    axes[index, 0].set_ylabel(str(np.round(val, 4)))
    axes[9, index].set_xlabel(str(np.round(val, 4)))
fig.subplots_adjust(wspace=0, hspace=0)
fig.savefig("Task4_all.png", dpi=300)
fig.show()
