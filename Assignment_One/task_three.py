from mnist import mnist_data
import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix

def ilogit(x, b):
    return 1/(1+np.exp(-b*x))
    
def ECDF(feat_v, ylabel):    
    classes = np.unique(ylabel)
    c1 = classes[0]
    c2 = classes[1]
    
    print("Class 1 is",c1)
    print("Class 2 is",c2)
    
    data1 = np.sort( feat_v[ (ylabel==c1).flatten() ])
    n1 = len(data1)
    data1 = data1.reshape((1,n1))
    cumf1 = np.reshape( ( np.arange(n1)+1 )/n1, (1,n1) )
    
    data2 = np.sort( feat_v[ (ylabel==c2).flatten() ])
    n2 = len(data2)
    data2 = data2.reshape((1,n2))
    cumf2 = np.reshape( ( np.arange(n2)+1 )/n1, (1,n2) )
    
    ECDF1 = np.concatenate( (data1,cumf1), axis=0)
    ECDF2 = np.concatenate( (data2,cumf2), axis=0)
    
    return ECDF1, ECDF2
    
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

def top_minus_bot_mass(obs):
    res = np.zeros(len(obs))
    tmb = np.concatenate([np.ones(8, np.int8),
                          np.ones(8, np.int8)*-1])
    rowsum = np.ones(16, np.int8)
    for i in range(len(obs)):
        res[i] = tmb@obs[i]@rowsum
    return res

def plot_hist(feat_v, ylabel):
    classes = np.unique(ylabel)
    c1 = classes[0]
    c2 = classes[1]
    data = [ feat_v[ (ylabel==c1).flatten() ],
             feat_v[ (ylabel==c2).flatten() ] ]
    prior_prob = list(map(lambda x: len(x)/len(feat_v), data))
    n = list(map(len, data))
    fig = plt.figure(0)
    ax = fig.add_subplot(111)
    epdf = ax.hist(data, label=[c1, c2])
    ax.set_title("Distribution of Feature Value")
    ax.set_ylabel("Frequency")
    ax.set_xlabel("Feature Value")
    ax.legend()
    
    return epdf, classes, prior_prob, n

def top_minus_bot_width(obs, b):
    res = np.zeros(len(obs))
    r1 = np.concatenate( [np.ones(5,np.int8),
                          np.zeros(11,np.int8)] ).reshape((1,16))
    r2 = np.concatenate( [np.zeros(11,np.int8),
                          np.ones(5,np.int8)] ).reshape((1,16))
    top_bot_sum = np.concatenate( (r1,r2), axis=0)
    for i in range(len(obs)):
        res[i] = (np.array([1,-1]) @\
           ((ilogit(top_bot_sum@(obs[i]+1), b)-0.5)*2) @ np.ones(16))
    return res

def predict(prior, epdf, classes, feat_v, n):
    y_hat = np.zeros(len(feat_v), np.int8)
    for i in range(len(feat_v)):
        floor = min(epdf[1])
        if feat_v[i] < floor:
            index = 0
        else:
            index = np.where(epdf[1]<=feat_v[i])[0][-1]
            if index == 10:
                index = 9
        post1 = epdf[0][0][index]/n[0]*prior[0]
        post2 = epdf[0][1][index]/n[1]*prior[1]
        if post1<post2:
            y_hat[i] = classes[1]
        else:
            y_hat[i] = classes[0]
    return y_hat

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues,
                          plotnum=0, **kwargs):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.

    Taken from the sklearn examples

    """
    plt.figure(plotnum)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)
    
    matrix = plt.imshow(cm, interpolation='nearest', cmap=cmap, **kwargs)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else '.0f'
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 verticalalignment="center",
                 fontsize = 12,
                 color="white" if matrix.norm(cm[i, j]) > 0.5 else "black")
    
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    
            
#==============================================================================
#  Main
#==============================================================================
def run_exp(num1, num2, feature, *args):
    x_train, y_train, x_test, y_test = load_data(5, 7)
    train_feat = feature(x_train, *args)
    test_feat = feature(x_test, *args)
    epdf, classes, prior, class_size = plot_hist(train_feat, y_train)
    y_hat_train = predict(prior, epdf, classes, train_feat, class_size)
    y_hat_test = predict(prior, epdf, classes, test_feat, class_size)
    train_cm = confusion_matrix(y_train, y_hat_train)
    test_cm = confusion_matrix(y_test, y_hat_test)
    plot_confusion_matrix(train_cm, classes, title="Training Set", plotnum = 1)
    plot_confusion_matrix(test_cm, classes, title="Test Set", plotnum = 2)
    