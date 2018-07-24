#!/usr/bin/env python
import os
import cv2
import glob
from sklearn.utils import shuffle
import numpy as np
import gc
import sys
from tools import *

np.set_printoptions(threshold=np.inf)

def read_image(filename, image_size, num_channels):
    """ Open a file, read the image and convert it to a optimum shape"""
    if num_channels == 1:
        image = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
    else:
        image = cv2.imread(filename)
    if image_size != len(image):
        image = cv2.resize(image, (image_size, image_size), 0, 0, cv2.INTER_LINEAR)
    image = image.astype(np.float32)
    # image = np.multiply(image, 1.0 / 255.0)
    image = np.where(image < 255, 0, 1)  # Improve the contrast of the dataset and transform 255 range to 0/1 values
    return (image)


def load_train(train_path, image_size, classes, shorter=0, num_channels=1):
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
    print("num_images : " + str(num_images))
    images = np.zeros((num_images, image_size, image_size), dtype=np.float32)
    labels = []
    img_names = []
    cls = []
    print('Going to read training images')
    k = 0
    for fields in classes:
        gc.collect()
        index = classes.index(fields)
        print('Now going to read {} files (Index: {})'.format(fields, index))
        path = os.path.join(train_path, fields, '*g')
        files = glob.glob(path)
        for i, fl in enumerate(files):
            if (i + 1) % 100 == 0 or (i + 1) == len(files):
                print("\t@ image " + str(i + 1) + "/" + str(len(files)))
            image = read_image(filename=fl, image_size=image_size, num_channels=num_channels)
            images[k] = image
            k += 1
            label = np.zeros(len(classes))
            label[index] = 1.0
            labels.append(label)
            flbase = os.path.basename(fl)
            img_names.append(flbase)
            cls.append(fields)
            if shorter != 0 and i + 1 >= shorter:
                break
    # Even if we want a single channel, we have to add a dimension to the array (dimension 1)
    if num_channels == 1:
        images = np.expand_dims(images, axis=3)
    labels = np.array(labels)
    print("\nImages array of shape : " + str(np.shape(images)))
    img_names = np.array(img_names)
    cls = np.array(cls)

    return images, labels, img_names, cls


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


def read_train_sets(train_path, image_size, classes, validation_size, shorter=0, num_channels=1,
                    dataset_save_dir_path=""):
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
        print("Try to load npy data in the memory...")
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
        print("Reloading the dataset...")
        images, labels, img_names, cls = load_train(train_path, image_size, classes, shorter=shorter,
                                                    num_channels=num_channels)
        images, labels, img_names, cls = shuffle(images, labels, img_names, cls)
        print("\nTry to write the dataset into the folder" + dataset_save_dir_path)
        try:
            createFolder(dataset_save_dir_path)
            np.save(lesArrayPath[0], images)
            np.save(lesArrayPath[1], labels)
            np.save(lesArrayPath[2], img_names)
            np.save(lesArrayPath[3], cls)
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
