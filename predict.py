#!/usr/bin/python3
# This project use the structure of the cv-tricks tutorial on Image Classification :
# http://cv-tricks.com/tensorflow-tutorial/training-convolutional-neural-network-for-image-classification/
# The scripts have been modified by Paul Asquin for a Room Classification project based on rooms 2D Maps
# Written by Paul Asquin - paul.asquin@gmail.com - Summer 2018


import dataset
import train
import numpy as np
import os
import sys, argparse
from tools import *
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR)

IMG_SIZE = 256
NUM_CHANNELS = 1
def main():

    print("Loading the image")
    dir_path = os.path.dirname(os.path.realpath(__file__))
    image_path = sys.argv[1]
    if image_path[0] != "/":
        image_path = dir_path + '/' + image_path
    image = np.array([dataset.read_image(filename=image_path, image_size=IMG_SIZE)], dtype=np.uint8)
    
    print("Shapping the image for the model input")
    # The input to the network is of shape [None image_size image_size num_channels]. Hence we reshape.
    x_batch = image.reshape(1, IMG_SIZE, IMG_SIZE, NUM_CHANNELS)

    print("Please choose the model to use : ")
    les_meta_path = locate_files(extension=".meta", path=os.getcwd(), dbName="meta")
    for i, meta_path in enumerate(les_meta_path):
        print("\n\n" + str(i) + " : " + str(meta_path))
        info_txt_path = str('/'.join(meta_path.split("/")[:-1]) + "/info.txt")
        try:
            with open(info_txt_path, 'r') as f:
                for line in f:
                    print("\t" + str(line.replace("\n", "")))
                print("")
        except FileNotFoundError:
            print("// No info.txt \n")
    model_num = int(input(">> "))

    try:
        meta_path = les_meta_path[model_num]
        model_dir_path = '/'.join(meta_path.split("/")[:-1]) + "/"
    except IndexError or TypeError:
        print("Wrong input")
        return -1

    print("Restoring the model", end="")
    sys.stdout.flush()
    sess = tf.Session()
    # Step-1: Recreate the network graph. At this step only graph is created.
    saver = tf.train.import_meta_graph(meta_path)
    # Step-2: Now let's load the weights saved using the restore method.
    saver.restore(sess, tf.train.latest_checkpoint(model_dir_path))
    graph = tf.get_default_graph()
    y_pred = graph.get_tensor_by_name("y_pred:0")
    print(" - Done")
    
    print("Feeding the image to the input")
    x = graph.get_tensor_by_name("x:0")
    y_true = graph.get_tensor_by_name("y_true:0")
    
    les_labels = []
    try:
        with open(model_dir_path + "labels.txt", 'r') as f:
            for line in f:
                label = line.replace("\n", "")
                if label != "":
                    les_labels.append(label)
    except Exception as e:
        les_labels = ['Bathroom', 'Bedroom', 'Kitchen', 'Living Room']
        print("Error openning labels.txt. We are going to use default values : " + str(les_labels))
        print("***\n" + str(e) + "\n***")
    print("Using labels : " + str(les_labels))
    
    y_test_images = np.zeros((1, len(les_labels)))
    ### Creating the feed_dict that is required to be fed to calculate y_pred
    feed_dict_testing = {x: x_batch, y_true: y_test_images}
    result = sess.run(y_pred, feed_dict=feed_dict_testing)
    print(result[0])
    # result is of this format [probabiliy_of_rose probability_of_sunflower]
    print("Prediction : ")
    for i in range(len(result[0])):
        print("\t" + les_labels[i] + " : " + str('{0:f}'.format(round(result[0][i]*100, 5))) + "%")


if __name__ == '__main__':
    main()
