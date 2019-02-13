from mnist import mnist_data
import numpy as np
from sklearn.metrics.pairwise import pairwise_distances as dist

x_train, y_train, x_test, y_test = mnist_data("data")


def clump_vectors(images, labels, dist_measure="euclidean"):
    """
    Task 1, calculate the mean of the image vectors as C, then calculate the radius of all labelled digits from C to
    get radius of each digit, and distances between the different centers
    :param images: Images in 16x16 format
    :param labels: Labels for the images
    :return: Centers, as well as radii for each center
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

    return center_lists, ri_list, list_of_center_distances


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


clump_vectors(x_train, y_train)

# Task 2

from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import itertools

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
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

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()


# Compute confusion matrix
cnf_matrix = confusion_matrix(y_test, y_pred)