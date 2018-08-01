#!/usr/bin/python3
# This project use the structure of the cv-tricks tutorial on Image Classification :
# http://cv-tricks.com/tensorflow-tutorial/training-convolutional-neural-network-for-image-classification/
# The scripts have been modified by Paul Asquin for a Room Classification project based on rooms 2D Maps
# Written by Paul Asquin - paul.asquin@gmail.com - Summer 2018

import os
import cv2
import glob
from sklearn.utils import shuffle
import numpy as np
import gc
import sys
from tools import *

np.set_printoptions(threshold=np.inf)

class DataSet(object):
    def __init__(self, images, labels, img_names, cls):
        self._num_examples = images.shape[0]

        self._images = images
        self._labels = labels
        self._img_names = img_names
        self._cls = cls
        self._epochs_done = 0
        self._index_in_epoch = 0

    @property
    def images(self):
        return self._images

    @property
    def labels(self):
        return self._labels

    @property
    def img_names(self):
        return self._img_names

    @property
    def cls(self):
        return self._cls

    @property
    def num_examples(self):
        return self._num_examples

    @property
    def epochs_done(self):
        return self._epochs_done

    def next_batch(self, batch_size):
        """Return the next `batch_size` examples from this data set."""
        start = self._index_in_epoch
        self._index_in_epoch += batch_size

        if self._index_in_epoch > self._num_examples:
            # After each epoch we update this
            self._epochs_done += 1
            start = 0
            self._index_in_epoch = batch_size
            assert batch_size <= self._num_examples
        end = self._index_in_epoch

        return self._images[start:end], self._labels[start:end], self._img_names[start:end], self._cls[start:end]


def read_image(filename, image_size):
    """ Open a file, read the image and convert it to a optimum shape"""
    image = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
    image = np.where(image > 30, 1, 0)  # Improve the contrast of the dataset and transform 255 range to 0/1 values
    image = image.astype(np.bool)
    return (image)


def load_train(train_path, image_size, classes, shorter=0, num_channels=1):
    """ Load the train dataset in memory """
    num_images = 0
    for fields in classes:
        path = os.path.join(train_path, fields, '*g')
        files = glob.glob(path)
        if shorter == 0:
            num_images += len(files)
        else:
            if len(files) < shorter:
                num_images += len(files)
            else:
                num_images += shorter
    print("\tTotal images number : " + str(num_images))
    images = np.zeros((num_images, image_size, image_size), dtype=np.bool)
    labels = np.zeros((num_images, len(classes)), dtype=np.bool)
    img_names = np.empty((num_images), dtype=object)
    cls = np.empty((num_images), dtype=object)
    print('\tGoing to read training images')
    k = 0
    for fields in classes:
        index = classes.index(fields)
        path = os.path.join(train_path, fields, '*g')
        if shorter != 0:
            files = glob.glob(path)[:shorter]
        else:
            files = glob.glob(path) 
        print('\tNow going to read {} files (Index: {}, Num: {})'.format(fields, index, len(files)))
        for i, fl in enumerate(files):
            if (i + 1) % 10 == 0 or (i + 1) == len(files):
                print("\t@ image " + str(i + 1) + "/" + str(len(files)) + " "*5, end="\r")
            images[k] = read_image(filename=fl, image_size=image_size)
            labels[k][index] = True
            img_names[k] = os.path.basename(fl)
            cls[k] = fields
            k += 1
        print("\n\t" + len(files) + " files loaded -> " + fields + " completed" + " "*10)
    # Even if we want a single channel, we have to add a dimension to the array (dimension 1)
    if num_channels == 1:
        images = np.expand_dims(images, axis=3)
    print("\nImages array of shape : " + str(np.shape(images)))
    
    return images, labels, img_names, cls


def read_train_sets(train_path, image_size, classes, validation_size, shorter=0, num_channels=1,
                    dataset_save_dir_path=""):
    """ An already used dataset is saved under .npy files for faster use. We use those files if they exists.
        Else, we just reload the whole dataset and save it for later use. """
    class DataSets(object):
        pass

    data_sets = DataSets()

    # Try to load the dataset from npy files. If not possible, reload the dataset
    lesArrayName = ['images', 'labels', 'img_names', 'cls']
    lesArrayPath = []
    for name in lesArrayName:
        lesArrayPath.append(dataset_save_dir_path + "/" + name + ".npy")
    reload_data = False
    for i, arrayPath in enumerate(lesArrayPath):
        if os.path.isfile(arrayPath) and os.path.getsize(arrayPath) < 10:
            print(lesArrayName + " npy file too small. Probably empty. Deleting")
            os.remove(arrayPath)
            reload_data = True
        elif not os.path.isfile(arrayPath):
            reload_data = True
    if not reload_data:
        print("Try to load npy data in the memory from " + str(dataset_save_dir_path))
        try:
            images = np.load(lesArrayPath[0])
            labels = np.load(lesArrayPath[1])
            img_names = np.load(lesArrayPath[2])
            cls = np.load(lesArrayPath[3])
        except KeyboardInterrupt:
            print("Stop loading")
            return 0
        except:
            reload_data = True
    if reload_data:
        print("Loading the dataset " + str(train_path))
        images, labels, img_names, cls = load_train(train_path, image_size, classes, shorter=shorter,
                                                    num_channels=num_channels)
        print("Shuffling")
        images, labels, img_names, cls = shuffle(images, labels, img_names, cls)
        print("\nTry to write the dataset into the folder" + dataset_save_dir_path)
        try:
            createFolder(dataset_save_dir_path)
            np.save(lesArrayPath[0], images)
            np.save(lesArrayPath[1], labels)
            np.save(lesArrayPath[2], img_names)
            np.save(lesArrayPath[3], cls)
            with open(DATASET_SAVE_DIR_PATH + "/labels.txt", "w") as f:
                for classe in classes:
                    f.write(classe + "\n")
            print("Done")
        except:
            print("Problem writing the dataset")

    if isinstance(validation_size, float):
        validation_size = int(validation_size * images.shape[0])

    data_sets.train = DataSet(images[validation_size:], labels[validation_size:], img_names[validation_size:], cls[validation_size:])
    data_sets.valid = DataSet(images[:validation_size], labels[:validation_size], img_names[:validation_size], cls[:validation_size])

    del images
    del labels
    del img_names
    del cls
    return data_sets
