import csv
import os
import pandas as pd
import numpy as np

def mnist_data(directory):
    """
    Obtains and returns the MNIST data set with labels in x_train, y_train, x_test, y_test format
    :return:
    """
    training_image_df = pd.read_csv(os.path.join(directory, "train_in.csv"), delimiter=",").values
    training_labels = pd.read_csv(os.path.join(directory, "train_out.csv")).values
    training_image_list = []

    for row in training_image_df:
        training_image_list.append(np.reshape(row, (16,16)))
    training_images = np.asarray(training_image_list)
    # Now do the same for the test
    test_image_df = pd.read_csv(os.path.join(directory, "test_in.csv"), delimiter=",").values
    test_labels = pd.read_csv(os.path.join(directory, "test_out.csv")).values
    test_image_list = []

    for row in test_image_df:
        test_image_list.append(np.reshape(row, (16,16)))
    test_images = np.asarray(test_image_list)

    return training_images, training_labels, test_images, test_labels