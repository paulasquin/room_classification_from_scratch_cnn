import dataset
import tensorflow as tf
import math
import random
import numpy as np
import os
import sys
import gc
import time

sys.path.append('../../')
from tools import *

# Adding Seed so that random initialization is consistent
from numpy.random import seed

seed(1)
from tensorflow import set_random_seed

set_random_seed(2)
# Free not allocated memory
gc.collect()

# Hide useless tensorflow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
HYPERPARAM_TXT_PATH = 'hyperparams.txt'

# === Model parameters ===
# HYPERPARAM
NUM_ITERATION = 4000
BATCH_SIZE = 32
LEARNING_RATE = 1e-4
SHORTER_DATASET_VALUE = 0
IMG_SIZE = 500
LES_NUM_FILTERS_CONV = [64, 64, 64, 128, 128, 128, 256, 256, 256, 512, 512, 512]
LES_CONV_FILTER_SIZE = [3] * len(LES_NUM_FILTERS_CONV)
FC_LAYER_SIZE = 128
DATASET_PATH = '../../JPG_Scannet_Aug'

# Load hyperparams from hyperparams.txt file if exists
if True and os.path.isfile(HYPERPARAM_TXT_PATH):
    print("Loading from " + HYPERPARAM_TXT_PATH)
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
DATASET_SAVE_DIR_PATH = \
    os.getcwd() + "/" + DATASET_PATH.split("/")[-1].lower() + "_" + str(IMG_SIZE) + "_" + str(SHORTER_DATASET_VALUE)
EXPORTS_DIR_PATH = '/media/nas/ScanNet/From_Scratch_CNN/cv-tricks/exports'
createFolder(EXPORTS_DIR_PATH)
EXPORTNUM_DIR_PATH = EXPORTS_DIR_PATH + "/export_" + str(getExportNumber(EXPORTS_DIR_PATH + "/"))
createFolder(EXPORTNUM_DIR_PATH)
MODEL_DIR_PATH = EXPORTNUM_DIR_PATH + "/model"
INFO_TXT_PATH = EXPORTNUM_DIR_PATH + "/info.txt"
CSV_TRAIN = EXPORTNUM_DIR_PATH + "/train.csv"

if len(LES_CONV_FILTER_SIZE) != len(LES_NUM_FILTERS_CONV):
    print("Convolutional layers params aren't the same length. Setting to 3*3")
    LES_CONV_FILTER_SIZE = [3] * len(LES_NUM_FILTERS_CONV)

# === Global variables ===
g_total_iterations = 0


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


def show_progress(epoch, feed_dict_train, feed_dict_validate, val_loss, session, accuracy, i, milestone=False, eta=0):
    acc = session.run(accuracy, feed_dict=feed_dict_train)
    val_acc = session.run(accuracy, feed_dict=feed_dict_validate)
    msg = "Training Epoch {0} --- Training Accuracy: {1:>6.1%}, Validation Accuracy: {2:>6.1%},  Validation Loss: {3:.3f}"
    prefix = ""
    suffix = ""
    if milestone:
        prefix = "\t"
        if eta != 0:
            h = int(eta/3600)
            min = int((eta - h * 3600)/60)

            suffix = ",  ETA : " + str(h) + "h" + str(min) + "m"
    else:
        prefix = "Saving model. "

    with open(CSV_TRAIN, 'a') as f:
        txt = str(i) + "\t" + str(epoch + 1) + "\t" + str(acc) + "\t" + str(val_acc) + "\t" + str(val_loss) + "\n"
        f.write(txt.replace('.', ','))

    print(prefix + msg.format(epoch + 1, acc, val_acc, val_loss) + suffix)


def train(num_iteration, session, data, cost, saver, accuracy, optimizer, x, y_true):
    global g_total_iterations
    tic = time.time()
    eta = 0
    #for i in range(g_total_iterations, g_total_iterations + num_iteration):
    for i in range(num_iteration):
        print("\t" + str(i) + " : [" + str(g_total_iterations) + ", " + str(
            g_total_iterations + num_iteration) + "]. Save every " + str(int(data.train.num_examples / BATCH_SIZE)))

        x_batch, y_true_batch, _, _ = data.train.next_batch(BATCH_SIZE)
        x_valid_batch, y_valid_batch, _, _ = data.valid.next_batch(BATCH_SIZE)

        feed_dict_tr = {x: x_batch,
                        y_true: y_true_batch}
        feed_dict_val = {x: x_valid_batch,
                         y_true: y_valid_batch}
        session.run(optimizer, feed_dict=feed_dict_tr)
        if i % int(data.train.num_examples / BATCH_SIZE) == 0:
            val_loss = session.run(cost, feed_dict=feed_dict_val)
            epoch = int(i / int(data.train.num_examples / BATCH_SIZE))
            show_progress(epoch, feed_dict_tr, feed_dict_val, val_loss, session=session, accuracy=accuracy, i=i)
            saver.save(session, MODEL_DIR_PATH)
        elif i % 5 == 0:
            if eta == 0:
                eta = (time.time() - tic)/5*(num_iteration-i)
            else:
                eta = int((eta/num_iteration + (time.time() - tic)/5)/2*(num_iteration - i))
            tic = time.time()
            val_loss = session.run(cost, feed_dict=feed_dict_val)
            epoch = int(i / int(data.train.num_examples / BATCH_SIZE))
            show_progress(epoch, feed_dict_tr, feed_dict_val, val_loss, session=session, accuracy=accuracy, i=i,
                          milestone=True, eta=eta)

    g_total_iterations += num_iteration


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


def main():
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
    print(classes)
    num_classes = len(classes)

    data = dataset.read_train_sets(
        DATASET_PATH,
        IMG_SIZE,
        classes,
        validation_size=VALIDATION_PERCENTAGE,
        shorter=SHORTER_DATASET_VALUE,
        num_channels=NUM_CHANNELS,
        dataset_save_dir_path=DATASET_SAVE_DIR_PATH
    )

    print("\nComplete reading input data. Will Now print a snippet of it")
    print("Number of files in Training-set:\t\t{}".format(len(data.train.labels)))
    print("Number of files in Validation-set:\t{}".format(len(data.valid.labels)))

    x = tf.placeholder(tf.float32, shape=[None, IMG_SIZE, IMG_SIZE, NUM_CHANNELS], name='x')

    # labels
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
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=lesLayers[-1],
                                                            labels=y_true)
    cost = tf.reduce_mean(cross_entropy)
    optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE).minimize(cost)
    correct_prediction = tf.equal(y_pred_cls, y_true_cls)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    session.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
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
        print("Exiting the training")
        pass

    session.close()
    gc.collect()  # Free not allocated memory


if __name__ == '__main__':
    main()
