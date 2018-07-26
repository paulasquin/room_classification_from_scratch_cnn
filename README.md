#room_classification_from_scratch_cnn
Project by [Paul Asquin](https://www.linkedin.com/in/paulasquin/) for Awabot - Summer 2018 paul.asquin@gmail.com  

# I.Introduction  
This repo is a part of the Room Classification Project. 
The aim of the Room Classification Project is to make an indoor mobile robot able to recognize a room using its 2D map. 
The output of the 2D map given should be "Kitchen", "Bedroom", "Batroom", etc.  

In order to achieve this goal, we have chosen to use Machine Learning techniques in order to obtain a powerfull recognition system with no hard-coded rules.  

As for every Machine Learning project, we need adapted datasets and a learning algorithm.  

Here is the overall architecture of the project :   
.  
├── room_classification_get_datasets  
├── room_classification_from_scratch_cnn  
├── room_classification_network_retrain  
├── Datasets (created be room_classification_get_datasets)  

Before comming to this repo, you should have installed the project datasets using _room\_classification\_get\_datasets_

# II. Goals and instructions
The goal of this repo is to build and train a Convolutional Neural Network in order to generate a model for the room classification [train.py](train.py), and use this model to make prediction on new given maps [predict.py](predict.py).
Those scripts are inspired by the work of the [cv-tricks](http://cv-tricks.com/) webite who developed the most reachable tutorial that I found on [Image CNN training](http://cv-tricks.com/tensorflow-tutorial/training-convolutional-neural-network-for-image-classification/).
In order to learn more about CNN operation, you can consult [this page](https://medium.com/@RaghavPrabhu/understanding-of-convolutional-neural-network-cnn-deep-learning-99760835f148) from [medium.com](https://medium.com).  

## 1. Build a network
For this code to build your own CNN model, you have to choose the hyperparameters that will define the model. 
For this, you will have to edit the file [hyperparams.txt](hyperparams.txt). Here are the pro and cons of each parameter change : 

**NUM_ITERATION**: number of training iterations.
\+ If too tall, the model still works but in the end will not learn anymore and performs unnecessary calculations
\- If it is too small, the model does not have time to reach its actual performance

**BATCH_SIZE**: The size of the image subpacket used for each train iteration.
\+ If too large, the necessary calculations and memory explode and the performance of the model decreases by loss of the generalization capacity.
\- If too small, gradient descents are less representative and performance calculations become noisy

**LEARNING_RATE**: learning speed, speed coefficient of the gradient descent.
\+ If too large, the gradient descent can lead to a divergence
\- If too low greatly slows the speed of calculation

**SHORTER_DATASET_VALUE** optional: limit the number of images per categories
\+ If the number of files used is too large, the demand in memory and calculation explodes.
\- If this number is too low, the model is lacking data to learn in a representative way

**IMG_SIZE**: size in pixels of images, with a native maximum of 500px
\+ If too big, the resolution of the images explodes the request in memory and calculation. Similarly, this feature may not be representative of the user application.
\- If too small, the resolution of the images no longer makes it possible to identify features on the cards

**DATASET_PATH**: used data set, for example ScanNet, ScanNet_Aug, ScanNet_Matterport_Tri_Aug ...
\+ If the dataset is too big or not specific enough, the model will not be able to learn from its features in a reasonable amount of time.
\- If the dataset is too small or its data too specific, the model will overfit to this dataset
\~ If the dataset contains too many errors, the results may become meaningless.

**LES_CONV_FILTER_SIZE**: list of the size of the convolution filters, that is to say size of the local area to study. See Figures 4 & 5 of medium.com
\+ If values are too large or if the list is too big, features will become invisible to the model
\- If values are too small or the list to small, the model will not be able to clear features effectively

**LES_NUM_FILTERS_CONV**: list of the number of filters per convolution layer, that is to say number of neurons per layer.
\+ If the values are too large, the memory and the necessary computing capacity grow enormously
\- If the values are too small, the model is not complex enough and can not learn data.

**FC_LAYER_SIZE**: size of the last Fully Connected layer (cf figure 9 in [this page](https://medium.com/@RaghavPrabhu/understanding-of-convolutional-neural-network-cnn-deep-learning-99760835f148))
\+ If the value is too large, the memory charge explodes
\- If the value is too low, the accuracy of the model falls considerably