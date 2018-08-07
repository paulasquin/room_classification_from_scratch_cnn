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
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR)

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
    """ Open a file, read the image and convert it to an optimum shape"""
    image = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
    if image_size != len(image):
        image = cv2.resize(image, (image_size, image_size), 0, 0, cv2.INTER_LINEAR)
    image = np.where(image > 30, 1, 0)  # Improve the contrast of the dataset and transform 255 range to 0/1 values
    image = image.astype(np.bool) # Using booleans for RAM optimization
    return (image)


def load_train(train_path, image_size, classes, shorter=0, num_channels=1):
    """ Load the train dataset in memory """
    print("\tLoading images paths")
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
    print("\tTotal number of images : " + str(num_images))

    print("\tInitializing arrays sizes", end="")
    sys.stdout.flush()
    images = np.zeros((num_images, image_size, image_size), dtype=np.bool)
    labels = np.zeros((num_images, len(classes)), dtype=np.bool)
    img_names = np.empty((num_images), dtype=object)
    cls = np.empty((num_images), dtype=object)
    print(" - Done")

    print('\tGoing to read training images')
    k = 0
    for fields in classes:
        index = classes.index(fields)
        path = os.path.join(train_path, fields, '*g')
        if shorter != 0:
            files = glob.glob(path)[:shorter]
        else:
            files = glob.glob(path) 
        print("\tReading " + fields + " files (Index: " + str(index) + ", Num: " + str(len(files)) + ")")
        for i, fl in enumerate(files):
            if (i + 1) % 10 == 0 or (i + 1) == len(files):
                print("\t@ image " + str(i + 1) + "/" + str(len(files)) + " "*5, end="\r")
            images[k] = read_image(filename=fl, image_size=image_size)
            labels[k][index] = True
            img_names[k] = os.path.basename(fl)
            cls[k] = fields
            k += 1
        print("\n\t" + str(len(files)) + " files loaded. " + fields + " completed" + " "*10)

    # Even if we want a single channel of black and white, or 1 and 0 (not 3 channels) 
    # we have to add the dimension 1 to the array
    if num_channels == 1:
        images = np.expand_dims(images, axis=3)
    print("\nImages array of shape : " + str(np.shape(images)))
    
    return images, labels, img_names, cls


def read_train_sets(train_path, image_size, classes, validation_size, shorter=0, num_channels=1,
                    dataset_save_dir_path=""):
    """ An already used dataset is saved under .npy files for faster use. We use those files if they exists.
        Otherwise, we just reload the whole dataset and save it for later use. """
    class DataSets(object):
        pass

    data_sets = DataSets()

    print("\nTrying to reload the dataset from npy files. If it's not possible, we will load the whole dataset")
    les_array_name = ['images', 'labels', 'img_names', 'cls']
    les_array_path = []
    refresh_data = False
    for name in les_array_name:
        les_array_path.append(dataset_save_dir_path + "/" + name + ".npy")
    for i, arrayPath in enumerate(les_array_path):
        if os.path.isfile(arrayPath) and os.path.getsize(arrayPath) < 10:
            print("\tThe " + les_array_name + " npy file is too small, probably empty due to an error. We remove it")
            os.remove(arrayPath)
            refresh_data = True
        elif not os.path.isfile(arrayPath):
            refresh_data = True
    
    if not refresh_data:
        print("\tLoading npy datas in the memory from " + str(dataset_save_dir_path))
        try:
            images = np.load(les_array_path[0])
            labels = np.load(les_array_path[1])
            img_names = np.load(les_array_path[2])
            cls = np.load(les_array_path[3])
        except KeyboardInterrupt:
            print("\tStop loading because of KeyboardInterrupt")
            return 0
        except Exception as e:
            print()
            print("\tProblem loading the dataset from npy datas. Going to reload the whole dataset.")
            print("***\n" + str(e) + "\n***")
            refresh_data = True

    if refresh_data:
        print("\tLoading the whole dataset from " + str(train_path))
        images, labels, img_names, cls = \
        load_train(
            train_path, 
            image_size, 
            classes, 
            shorter=shorter,
            num_channels=num_channels
        )
        print("Shuffling the datas")
        images, labels, img_names, cls = shuffle(images, labels, img_names, cls)
        print("Trying to write the dataset in npy files into the folder : \n" + dataset_save_dir_path, end="")
        sys.stdout.flush()
        try:
            createFolder(dataset_save_dir_path)
            np.save(les_array_path[0], images)
            np.save(les_array_path[1], labels)
            np.save(les_array_path[2], img_names)
            np.save(les_array_path[3], cls)
            print(" - Done")
        except Exception as e:
            print("\nProblem writing the dataset. Skiping.")
            print("***\n" + str(e) + "\n***")

    if isinstance(validation_size, float):
        validation_size = int(validation_size * images.shape[0])
    print("\tSplitting the dataset into train and validation sets", end="")
    sys.stdout.flush()
    data_sets.train = DataSet(images[validation_size:], labels[validation_size:], img_names[validation_size:], cls[validation_size:])
    data_sets.valid = DataSet(images[:validation_size], labels[:validation_size], img_names[:validation_size], cls[:validation_size])
    print(" - Done")
    del images
    del labels
    del img_names
    del cls
    return data_sets
