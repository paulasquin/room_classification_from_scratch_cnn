#!/usr/bin/python3
# This project use the structure of the cv-tricks tutorial on Image Classification :
# http://cv-tricks.com/tensorflow-tutorial/training-convolutional-neural-network-for-image-classification/
# The scripts have been modified by Paul Asquin for a Room Classification project based on rooms 2D Maps
# Written by Paul Asquin - paul.asquin@gmail.com - Summer 2018

import dataset
import math
import random
import numpy as np
import os
import sys
import gc
import time
import shutil
from tools import *
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR)
# Adding Seed so that random initialization is consistent
from numpy.random import seed

seed(1)
from tensorflow import set_random_seed

set_random_seed(2)
# Free not allocated memory
gc.collect()

# Hide useless tensorflow warnings

HYPERPARAM_TXT_PATH = 'hyperparams.txt'

# === Model parameters ===
# HYPERPARAM
NUM_ITERATION = 4000
BATCH_SIZE = 128
LEARNING_RATE = 1e-4
SHORTER_DATASET_VALUE = 0
IMG_SIZE = 256
LES_NUM_FILTERS_CONV = [64, 64, 64, 128, 128, 128, 256, 256, 256, 512, 512, 512]
LES_CONV_FILTER_SIZE = [3] * len(LES_NUM_FILTERS_CONV)
FC_LAYER_SIZE = 128
DATASET_PATH = "../Datasets/JPG"

# Load hyperparams from hyperparams.txt file if exists
if os.path.isfile(HYPERPARAM_TXT_PATH):
    print("Loading hyperparameters from " + HYPERPARAM_TXT_PATH)
    with open(HYPERPARAM_TXT_PATH, 'r') as f:
        for line in f:
            if 'NUM_ITERATION' in line:
                NUM_ITERATION = int(line.split(" = ")[-1])
            elif 'BATCH_SIZE' in line:
                BATCH_SIZE = int(line.split(" = ")[-1])
            elif 'LEARNING_RATE' in line:
                LEARNING_RATE = float(line.split(" = ")[-1])
            elif 'SHORTER_DATASET_VALUE' in line:
                SHORTER_DATASET_VALUE = int(line.split(" = ")[-1])
            elif 'IMG_SIZE' in line:
                IMG_SIZE = int(line.split(" = ")[-1])
            elif 'FC_LAYER_SIZE' in line:
                FC_LAYER_SIZE = int(line.split(" = ")[-1])
            elif 'DATASET_PATH' in line:
                DATASET_PATH = line.split(" = ")[-1].replace("\n", '').replace("'", "")
            elif 'LES_NUM_FILTERS_CONV' in line:
                LES_NUM_FILTERS_CONV = \
                    list(map(int, line.replace('LES_NUM_FILTERS_CONV = [', '').replace(']', '').split(', ')))
            elif 'LES_CONV_FILTER_SIZE' in line:
                LES_CONV_FILTER_SIZE = \
                    list(map(int, line.replace('LES_CONV_FILTER_SIZE = [', '').replace(']', '').split(', ')))

NUM_CHANNELS = 1
VALIDATION_PERCENTAGE = 0.2  # 20% of the data will automatically be used for validation
DATASET_SAVE_DIR_PATH = os.getcwd() + "/" + DATASET_PATH.split("/")[-1].lower() + "_" + str(IMG_SIZE) + "_" + str(SHORTER_DATASET_VALUE)
EXPORTS_DIR_PATH = os.getcwd() + '/exports'
createFolder(EXPORTS_DIR_PATH)
EXPORTNUM_DIR_PATH = EXPORTS_DIR_PATH + "/export_" + str(getExportNumber(EXPORTS_DIR_PATH + "/"))
createFolder(EXPORTNUM_DIR_PATH)
MODEL_DIR_PATH = EXPORTNUM_DIR_PATH + "/model"
INFO_TXT_PATH = EXPORTNUM_DIR_PATH + "/info.txt"
CSV_TRAIN = EXPORTNUM_DIR_PATH + "/train.csv"

if len(LES_CONV_FILTER_SIZE) != len(LES_NUM_FILTERS_CONV):
    print("Convolutional layers params aren't the same length. Setting to 3*3")
    LES_CONV_FILTER_SIZE = [3] * len(LES_NUM_FILTERS_CONV)

class ConvolutionLayer:
    inputt = 0
    num_input_channels = 0
    conv_filter_size = 0
    num_filters = 0
    layer = 0

    def __init__(self, inputt, num_input_channels, conv_filter_size, num_filters):
        self.inputt = inputt
        self.num_input_channels = num_input_channels
        self.conv_filter_size = conv_filter_size
        self.num_filters = num_filters
        self.layer = create_convolutional_layer(
            input=inputt,
            num_input_channels=num_input_channels,
            conv_filter_size=conv_filter_size,
            num_filters=num_filters
        )


def create_weights(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.05))


def create_biases(size):
    return tf.Variable(tf.constant(0.05, shape=[size]))


def create_convolutional_layer(input,
                               num_input_channels,
                               conv_filter_size,
                               num_filters):
    # We shall define the weights that will be trained using create_weights function.

    weights = create_weights(shape=[conv_filter_size, conv_filter_size, num_input_channels, num_filters])
    # We create biases using the create_biases function. These are also trained.
    biases = create_biases(num_filters)

    # Creating the convolutional layer
    layer = tf.nn.conv2d(input=input,
                         filter=weights,
                         strides=[1, 1, 1, 1],
                         padding='SAME')

    layer += biases

    # We shall be using max-pooling.
    layer = tf.nn.max_pool(value=layer,
                           ksize=[1, 2, 2, 1],
                           strides=[1, 2, 2, 1],
                           padding='SAME')

    # Output of pooling is fed to Relu which is the activation function for us.
    layer = tf.nn.relu(layer)

    return layer


def create_flatten_layer(layer):
    # We know that the shape of the layer will be [batch_size img_size img_size num_channels]
    # But let's get it from the previous layer.
    layer_shape = layer.get_shape()

    # Number of features will be img_height * img_width* num_channels.
    # But we shall calculate it in place of hard-coding it.
    num_features = layer_shape[1:4].num_elements()

    # Now, we Flatten the layer so we shall have to reshape to num_features
    layer = tf.reshape(layer, [-1, num_features])

    return layer


def create_fc_layer(input, num_inputs, num_outputs, use_relu=True):
    # Let's define trainable weights and biases.
    weights = create_weights(shape=[num_inputs, num_outputs])
    biases = create_biases(num_outputs)

    # Fully connected layer takes input x and produces wx+b.Since, these are matrices, we use matmul function in Tensorflow
    layer = tf.matmul(input, weights) + biases
    if use_relu:
        layer = tf.nn.relu(layer)

    return layer


def show_progress(epoch, feed_dict_train, feed_dict_validate, val_loss, session, accuracy, i, milestone=False, time_left=0):
    acc = session.run(accuracy, feed_dict=feed_dict_train)
    val_acc = session.run(accuracy, feed_dict=feed_dict_validate)
    msg = "\tTraining Epoch {0}\n\t\tTraining Accuracy\t{1:>6.1%}\n\t\tValidation Accuracy\t{2:>6.1%}\n\t\tValidation Loss\t\t{3:.3f}"
    suffix = ""
    time_left_msg = ""
    if milestone:
        if time_left != 0:
            h = int(time_left / 3600)
            min = int((time_left - h * 3600) / 60)
            time_left_msg = "\n\t\tTime Left\t\t~ " + str(h) + "h" + str(min) + "m"
    else:
        print("\tSaving the model." + " "*20)

    with open(CSV_TRAIN, 'a') as f:
        txt = str(i) + "\t" + str(epoch + 1) + "\t" + str(acc) + "\t" + str(val_acc) + "\t" + str(val_loss) + "\n"
        f.write(txt.replace('.', ','))

    print(msg.format(epoch + 1, acc, val_acc, val_loss) + time_left_msg)


def train(num_iteration, session, data, cost, saver, accuracy, optimizer, x, y_true):
    print("Start the training. Saving every " + str(int(data.train.num_examples / BATCH_SIZE)) + " iterations.")
    tic = time.time()
    time_left = 0
    for i in range(num_iteration):
        print(str(i+1) + "/" + str(num_iteration) + " - Loading the batch" + " "*20, end="\r")
        x_batch, y_true_batch, _, _ = data.train.next_batch(BATCH_SIZE)
        x_valid_batch, y_valid_batch, _, _ = data.valid.next_batch(BATCH_SIZE)
        feed_dict_tr = {x: x_batch, y_true: y_true_batch}

        print(str(i+1) + "/" + str(num_iteration) + " - Running the optimization" + " "*20, end="\r")
        session.run(optimizer, feed_dict=feed_dict_tr)
        feed_dict_val = {x: x_valid_batch, y_true: y_valid_batch}
        print(str(i+1) + "/" + str(num_iteration) + " - Completed" + " "*20)

        if i % int(data.train.num_examples / BATCH_SIZE) == 0:
            val_loss = session.run(cost, feed_dict=feed_dict_val)
            epoch = int(i / int(data.train.num_examples / BATCH_SIZE))
            show_progress(epoch, feed_dict_tr, feed_dict_val, val_loss, session=session, accuracy=accuracy, i=i)
            saver.save(session, MODEL_DIR_PATH)
        elif i % 5 == 0:
            if time_left == 0:
                time_left = (time.time() - tic) / 5 * (num_iteration - i)
            else:
                time_left = int((time_left / num_iteration + (time.time() - tic) / 5) / 2 * (num_iteration - i))
            tic = time.time()
            val_loss = session.run(cost, feed_dict=feed_dict_val)
            epoch = int(i / int(data.train.num_examples / BATCH_SIZE))
            show_progress(
                epoch, 
                feed_dict_tr, 
                feed_dict_val, 
                val_loss, 
                session=session, 
                accuracy=accuracy, 
                i=i,
                milestone=True, 
                time_left=time_left
            )


def init():
    with open(INFO_TXT_PATH, 'w') as f:
        txt = "NUM_ITERATION = " + str(NUM_ITERATION) + \
              "\nBATCH_SIZE = " + str(BATCH_SIZE) + \
              "\nLEARNING_RATE = " + str(LEARNING_RATE) + \
              "\nSHORTER_DATASET_VALUE = " + str(SHORTER_DATASET_VALUE) + \
              "\nIMG_SIZE = " + str(IMG_SIZE) + \
              "\nDATASET_PATH = " + str(DATASET_PATH) + \
              "\nLES_CONV_FILTER_SIZE = " + str(LES_CONV_FILTER_SIZE) + \
              "\nLES_NUM_FILTERS_CONV = " + str(LES_NUM_FILTERS_CONV) + \
              "\nFC_LAYER_SIZE = " + str(FC_LAYER_SIZE)
        f.write(txt)
    with open(CSV_TRAIN, 'w') as f:
        f.write("Iteration\tEpoch\tTraining Accuracy\tValidation Accuracy\tValidation Loss\n")


def cleanExports():
    """ Remove folder with no model written """
    les_folders = os.listdir(EXPORTS_DIR_PATH)
    for folder in les_folders:
        if "export" in folder and not os.path.isfile(EXPORTS_DIR_PATH + "/" + folder + "/model.meta"):
            print("Removing " + EXPORTS_DIR_PATH + "/" + folder)
            shutil.rmtree(EXPORTS_DIR_PATH + "/" + folder)    


def main():
    cleanExports()
    gc.collect()
    init()
    session = tf.Session()
    # Prepare input data
    classes = os.listdir(DATASET_PATH)
    i = 0
    while i < len(classes):
        if "." in classes[i]:
            classes.pop(i)
        else:
            i += 1
    classes.sort()
    print("Labels : " + str(classes))
    num_classes = len(classes)
    data = dataset.read_train_sets(
        DATASET_PATH,
        IMG_SIZE,
        classes,
        validation_size=VALIDATION_PERCENTAGE,
        shorter=SHORTER_DATASET_VALUE,
        dataset_save_dir_path=DATASET_SAVE_DIR_PATH
    )
    print("Complete reading input data.")
    print("Number of files in Training-set : " + str(len(data.train.labels)))
    print("Number of files in Validation-set : " + str(len(data.valid.labels)))

    print("\nBuilding the model", end="")
    sys.stdout.flush()
    x = tf.placeholder(tf.float32, shape=[None, IMG_SIZE, IMG_SIZE, NUM_CHANNELS], name='x')
    y_true = tf.placeholder(tf.float32, shape=[None, num_classes], name='y_true')
    y_true_cls = tf.argmax(y_true, axis=1)
    lesLayers = []
    # Adding Convolutional layers
    for i in range(len(LES_NUM_FILTERS_CONV)):
        if i == 0:
            inputt = x
            num_input_channels = NUM_CHANNELS
        else:
            inputt = lesLayers[-1].layer
            num_input_channels = lesLayers[-1].num_filters

        lesLayers.append(ConvolutionLayer(inputt, num_input_channels, LES_CONV_FILTER_SIZE[i], LES_NUM_FILTERS_CONV[i]))
    # Adding flatten layer
    lesLayers.append(create_flatten_layer(lesLayers[-1].layer))
    # Adding fully connected layers
    lesLayers.append(
        create_fc_layer(
            input=lesLayers[-1],
            num_inputs=lesLayers[-1].get_shape()[1:4].num_elements(),
            num_outputs=FC_LAYER_SIZE,
            use_relu=True
        )
    )
    lesLayers.append(
        create_fc_layer(
            input=lesLayers[-1],
            num_inputs=FC_LAYER_SIZE,
            num_outputs=num_classes,
            use_relu=False
        )
    )
    y_pred = tf.nn.softmax(lesLayers[-1], name='y_pred')
    y_pred_cls = tf.argmax(y_pred, axis=1)
    session.run(tf.global_variables_initializer())
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
        logits=lesLayers[-1],
        labels=y_true
    )
    cost = tf.reduce_mean(cross_entropy)
    optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE).minimize(cost)
    correct_prediction = tf.equal(y_pred_cls, y_true_cls)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    session.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    print(" - Done")

    print("Saving the labels", end="")
    sys.stdout.flush()
    with open(EXPORTNUM_DIR_PATH + "/labels.txt", "w") as f:
        for classe in classes:
            f.write(classe + "\n")
    print(" - Done")

    try:
        train(
            num_iteration=NUM_ITERATION,
            session=session,
            data=data,
            cost=cost,
            saver=saver,
            accuracy=accuracy,
            optimizer=optimizer,
            x=x,
            y_true=y_true
        )
    except KeyboardInterrupt:
        print("Exiting the training due to KeyboardInterrupt")
        pass
    print("Clean exiting", end="")
    sys.stdout.flush()
    session.close()
    gc.collect()
    print(" - Done\n\n")
    return 0

if __name__ == '__main__':
    main()
