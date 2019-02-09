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

    #TODO What does the Ci, ni mean in the instructions? Not clear what he is saying

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


clump_vectors(x_train, y_train)