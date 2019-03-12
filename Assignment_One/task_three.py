from mnist import mnist_data
import numpy as np
from matplotlib import pyplot as plt

def load_data(num1, num2):
    x_train, y_train, x_test, y_test = mnist_data("data")
    
    train_bool = ((y_train == num1)|(y_train == num2)).flatten()
    test_bool = ((y_test == num1)|(y_test == num2)).flatten()
    
    x_train = x_train[train_bool]
    y_train = y_train[train_bool]
    x_test = x_test[test_bool]
    y_test = y_test[test_bool]
    
    print("Train set:", sum(y_train==num1)[0], "observations of number", num1)
    print("Train set:", sum(y_train==num2)[0], "observations of number", num2)
    print("Test set:", sum(y_test==num1)[0], "observations of number", num1)
    print("Test set:", sum(y_test==num2)[0], "observations of number", num2)
    
    return x_train, y_train, x_test, y_test

def top_minus_bot(ndarray):
    res = np.zeros(len(ndarray))
    tmb = np.concatenate([np.ones(8, np.int8),
                          np.ones(8, np.int8)*-1])
    rowsum = np.ones(16, np.int8)
    for i in range(len(ndarray)):
        res[i] = tmb@ndarray[i]@rowsum
    return res

def plot_hist(feat_v, ylabel):
    classes = np.unique(ylabel)
    c1 = classes[0]
    c2 = classes[1]
    data = [ feat_v[ (ylabel==c1).flatten() ],
             feat_v[ (ylabel==c2).flatten() ] ]
        
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.hist(data, label=[c1, c2])
    ax.set_title("Distribution of Feature Value")
    ax.set_ylabel("Frequency")
    ax.set_xlabel("Feature Value")
    ax.legend()

