from mnist import mnist_data
import numpy as np
from sklearn.metrics.pairwise import pairwise_distances as dist
from myplot import heatmap

x_train, y_train, x_test, y_test = mnist_data("data")


def clump_vectors(images, labels, dist_measure="euclidean"):
    """
    Task 1, calculate the mean of the image vectors as C, then calculate the radius of all labelled digits from C to
    get radius of each digit, distances between the different centers, and the number of points that belong to each C
    :param images: Images in 16x16 format
    :param labels: Labels for the images
    :return: Centers, radii for each center, distances between centers, and number of elements belonging to each center
    """

    # TODO What does the Ci, ni mean in the instructions? Not clear what he is saying

    center_lists = []
    print(center_lists)
    for i in range(10):
        center_i = np.zeros(256)
        num_elements = 0
        # Inefficient way, go through the whole thing about 10 times
        for index, element in enumerate(labels):
            if element == i:
                center_i += images[index].reshape(256)
                num_elements += 1
        # get mean now
        center_i /= num_elements
        print(center_i.shape)
        center_lists.append(center_i)

    # Now calculate the ri and pairwise distances for this
    # TODO Check this, this is calculating over 0-9 points, not all points
    ri_list = []
    for i in range(10):
        center_i = center_lists[i]
        max_distance = 0
        for index, element in enumerate(labels):
            if element == i:

                thing = np.vstack((center_i, images[index].reshape(256)))
                print(thing.shape)
                distance = dist(thing, metric=dist_measure)[0][1]
                print(distance)
                if distance > max_distance:
                    max_distance = distance
        ri_list.append(max_distance)

    # TODO Count number with elements that have label Ci which is then ni
    num_elements_per_label = np.zeros((10))
    for label in labels:
        num_elements_per_label[label[0]] += 1


    # Finally calculating distances between centers
    list_of_center_distances = []
    for i in range(10):
        center_i_dist = []
        center_i = center_lists[i]
        for center in center_lists:
            thing = np.vstack((center_i, center))
            print(thing.shape)
            distance = dist(thing, metric=dist_measure)[0][1]
            center_i_dist.append(distance)
        list_of_center_distances.append(center_i_dist)

    list_of_center_distances = np.asarray(list_of_center_distances)
    ri_list = np.asarray(ri_list)
    center_lists = np.asarray(center_lists)

    print(list_of_center_distances)
    
    overlap = np.tile(ri_list, (10, 1)) + np.tile(ri_list, (10, 1)).T \
        - list_of_center_distances

    return center_lists, ri_list, list_of_center_distances, \
        num_elements_per_label, overlap


def classify_on_centers(image, centers, dist_measure="euclidean"):
    """
    Given a 16x16 image, get pairwise distance to calculate from centers
    :param image:
    :param centers:
    :return: Calculated digit
    """

    # Change to vector
    image.reshape((256))
    min_center = np.Inf
    min_center_index = -1
    for index, center in enumerate(centers):
        thing = np.vstack((image, center))
        distance = dist(thing, metric=dist_measure)[0][1]
        if distance < min_center:
            min_center = distance
            min_center_index = index

    return min_center_index

center_lists, ri_list, list_of_center_distances, num_elements_per_label, \
    overlap = clump_vectors(x_train, y_train)

heatmap(overlap, np.arange(10), np.arange(10), valfmt="{x:.0f}",
        textcolors=['white', 'black'], vmax = 25, cmap = 'hot')

#==============================================================================
# the heat map shows the overlap between the clusters in Euclidean distance;
# e.g. the (i, j)th value = r_i + r_j - d_ij, the diagonal gives the diameter
# of each cluster. The results show significant overlap, which is often more
# than half the diameter of the corresponding clusters, hence we would expect
# classification performance to be low. One limitation of this overlap matrix
# is that it is completely dependent on the extremum of each cluster, hence 
# does not provide any information about the distribution of values within the
# clusters. A more informative measure might be to plot the overlap matrix for
# the first 75% of values from the center.
#==============================================================================

# Task 2
# Training set prediction
yhat_train = np.zeros(len(x_train))
for i in range(len(x_train)):
    yhat_train[i] = classify_on_centers(x_train[i].reshape(256), center_lists)

# Test set prediction
yhat_test = np.zeros(len(x_test))
for i in range(len(x_test)):
    yhat_test[i] = classify_on_centers(x_test[i].reshape(256), center_lists)
    

from sklearn.metrics import confusion_matrix
import itertools
from matplotlib import pyplot as plt

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues,
                          sums = False, **kwargs):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.

    Taken from the sklearn examples

    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)
    
    if sums:    
        plt.subplot(221)
    
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
                 fontsize = 8,
                 color="white" if matrix.norm(cm[i, j]) > 0.5 else "black")
    
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    
    if sums:
        fmtsums = '.2f'
        
        plt.subplot(222)
        rec = np.reshape(np.diag(cm)/sum(cm), (10,1))
        
        recfig = plt.imshow(rec, interpolation='nearest', cmap=cmap)
        plt.title("Recall")
        plt.xticks([])
        plt.yticks(np.arange(len(classes)))
        
        for i, j in itertools.product(range(rec.shape[0]), range(rec.shape[1])):
            plt.text(j, i, format(rec[i, j], fmtsums),
                     horizontalalignment="center",
                     verticalalignment="center",
                     fontsize = 8,
                     color="white" if recfig.norm(rec[i, j]) > 0.5 else "black")
        
        plt.subplot(223)
        prec = np.reshape(np.diag(cm)/sum(cm, 1), (1,10))
        
        precfig = plt.imshow(prec, interpolation='nearest', cmap=cmap)
        plt.title("Precision")
        plt.xticks(np.arange(len(classes)))
        plt.yticks([])
        
        for i, j in itertools.product(range(prec.shape[0]), range(prec.shape[1])):
            plt.text(j, i, format(prec[i, j], fmtsums),
                     horizontalalignment="center",
                     verticalalignment="center",
                     fontsize = 8,
                     color="white" if precfig.norm(prec[i, j]) > 0.5 else "black")
            
# Compute and plot confusion matrix
train_cm = confusion_matrix(y_train, yhat_train)
test_cm = confusion_matrix(y_test, yhat_test)

plot_confusion_matrix(train_cm, range(10),
                      title = "Confusion Matrix - Training Set",
                      vmax = 30, sums = True)
plot_confusion_matrix(test_cm, range(10),
                      title = "Confusion Matrix - Test Set",
                      vmax = 30, sums = True)

# Manually compare y_test and yhat_test
np.asarray([np.reshape(y_test, 999), yhat_test])

plot_confusion_matrix(overlap, range(10), title = "Overlap Matrix", vmax = 25)



