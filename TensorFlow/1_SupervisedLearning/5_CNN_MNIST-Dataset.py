#!/usr/bin/env python
# coding: utf-8

# In this section, we will use the famous [MNIST Dataset](http://yann.lecun.com/exdb/mnist/) to build two Neural
# Networks capable to perform handwritten digits classification. The first Network is a simple Multi-layer Perceptron
# (MLP) and the second one is a Convolutional Neural Network (CNN from now on). In other words, when given an input our
# algorithm will say, with some associated error, what type of digit this input represents.
#   - What is Deep Learning?
#   - Load MNIST dataset
#   - 1st part: classify MNIST using a simple softmax regression model
#   - 2nd part: Deep Learning with CNN applied on MNIST
#       - Build the CNN
#       - Define functions and train the model
#       - Evaluate the model

import tensorflow as tf
from input_data import read_data_sets
# input_data.py was downloaded from the github repository of tensorflow
import numpy as np
from utils1 import tile_raster_images
# utils1.py was downloaded from http://deeplearning.net/tutorial/code/utils.py
import matplotlib.pyplot as plt
from PIL import Image


# -----------------
# #### What is Deep Learning?

# Brief Theory: Deep learning (also known as deep structured learning, hierarchical learning or deep machine learning)
# is a branch of machine learning based on a set of algorithms that attempt to model high-level abstractions in data by
# using multiple processing layers, with complex structures or otherwise, composed of multiple non-linear
# transformations.

# - Defining the term "Deep": in this context, deep means that we are studying a Neural Network which has several hidden
# layers (more than one), no matter what type (convolutional, pooling, normalization, fully-connected etc). The most
# interesting part is that some papers noticed that Deep Neural Networks with the right architectures/hyper-parameters
# achieve better results than shallow Neural Networks with the same computational power (e.g. number of neurons or
# connections).

# - Defining "Learning": In the context of supervised learning, digits recognition in our case, the learning part
# consists of a target/feature which is to be predicted using a given set of observations with the already known final
# prediction (label). In our case, the target will be the digit (0, 1, 2, 3, 4, 5, 6, 7, 8, 9) and the observations are
# the intensity and relative position of the pixels. After some training, it is possible to generate a "function" that
# map inputs (digit image) to desired outputs(type of digit). The only problem is how well this map operation occurs.
# While trying to generate this "function", the training process continues until the model achieves a desired level of
# accuracy on the training data.

# In this tutorial, we first classify MNIST using a simple Multi-layer perceptron and then, in the second part, we use
# deeplearning to improve the accuracy of our results.


# -----------------
# #### Load MNIST dataset
print('\n\n ---> Load the MNIST datset.')

# ## What is MNIST?
# According to LeCun's website, the MNIST is a: "database of handwritten digits that has a training set of 60,000
# examples, and a test set of 10,000 examples. It is a subset of a larger set available from MNIST. The digits have been
# size-normalized and centered in a fixed-size image". If you are not familiar with the MNIST dataset, please consider
# to read more about it: http://yann.lecun.com/exdb/mnist/

# It's very important to notice that MNIST is a high optimized data-set and it does not contain images. You will need
# to build your own code if you want to see the real digits. Another important side note is the effort that the authors
# invested on this data-set with normalization and centering operations.

# We can import the MNIST dataset using TensorFlow built-in feature.
mnist = read_data_sets("Data/MNIST_data/", one_hot=True)
batch_size = 50
# Similarly, we can do:
# mnist = tf.keras.datasets.mnist.load_data()
# but in this way it will be a tuple where mnist[0][0] are the training samples and mnist[0][1] the training labels,
# while mnist[1][0] are the test samples and mnist[1][1] the test labels.
# Another way is to use:
# import tensorflow_datasets as tfds
# mnist, info = tfds.load('mnist', with_info=True)
# but in this case it will be a dictionary where mnist['train'] is an object of training data and mnist['test'] an
# object of testing data.

# Note: the one-hot = True argument only means that, in contrast to Binary representation, the labels will be presented
# in a way that to represent a number N, the N^{th} bit is 1 while the the other bits are 0. For example, zero and five
# in a binary code would be:
#
# Number representation:    0
# Binary encoding:        [2^5]  [2^4]   [2^3]   [2^2]   [2^1]   [2^0]
# Array/vector:             0      0       0       0       0       0
#
# Number representation:    5
# Binary encoding:        [2^5]  [2^4]   [2^3]   [2^2]   [2^1]   [2^0]
# Array/vector:             0      0       0       1       0       1

# Using a different notation, the same digits using one-hot vector representation can be show as:
#
# Number representation:    0
# One-hot encoding:        [5]   [4]    [3]    [2]    [1]   [0]
# Array/vector:             0     0      0      0      0     1
#
# Number representation:    5
# One-hot encoding:        [5]   [4]    [3]    [2]    [1]    [0]
# Array/vector:             1     0      0      0      0      0

# ## Understanding the imported data.
# The imported data can be divided as follow:
#
# --> Training (mnist.train) >>  Use the given dataset with inputs and related outputs for training of NN. In our case,
# if you give an image that you know that represents a "nine", this set will tell the neural network that we expect a
# "nine" as the output.
#         - 55,000 data points
#         - mnist.train.images for inputs
#         - mnist.train.labels for outputs
#
# --> Validation (mnist.validation) >> The same as training, but now the data is used to generate model properties
# (classification error, for example) and from this, tune parameters like the optimal number of hidden units or
# determine a stopping point for the back-propagation algorithm.
#         - 5,000 data points
#         - mnist.validation.images for inputs
#         - mnist.validation.labels for outputs
#
# --> Test (mnist.test) >> the model does not have access to this information prior to the testing phase. It is used
# to evaluate the performance and accuracy of the model against "real life situations". No further optimization beyond
# this point.
#         - 10,000 data points
#         - mnist.test.images for inputs
#         - mnist.test.labels for outputs


# -----------------
# #### 1st part: classify MNIST using a simple model.
print('\n\n ---> Softmax Regression with Multi-Layer Perceptron for MNIST classification.')
# We are going to create a simple Multi-layer perceptron, a simple type of Neural Network, to perform classification
# tasks on the MNIST digits dataset.

# ## Create an interactive session
# You have two basic options when using TensorFlow to run your code:
# - [Build graphs and run session] Do all the set-up and THEN execute a session to evaluate tensors and run operations.
# - [Interactive session] create your coding and run on the fly.
# For this first part, we will use the interactive session that is more suitable for environments like Jupyter notebook.
sess = tf.compat.v1.InteractiveSession()

# ## Create placeholders
# It is a best practice to create placeholders before variable assignments when using TensorFlow. Here we'll create
# placeholders for inputs ("Xs") and outputs ("Ys").
tf.compat.v1.disable_eager_execution()
x = tf.compat.v1.placeholder(tf.float32, shape=[None, 784])
y_ = tf.compat.v1.placeholder(tf.float32, shape=[None, 10])

# Placeholder 'X': represents the "space" allocated input or the images
#    - Each input has 784 pixels distributed by a 28 width x 28 height matrix
#    - The 'shape' argument defines the tensor size by its dimensions.
#    - 1st dimension = None. Indicates that the batch size, can be of any size.
#    - 2nd dimension = 784. Indicates the number of pixels on a single flattened MNIST image.

# Placeholder 'Y': represents the final output or the labels.
#    - 10 possible classes (0, 1, 2, 3, 4, 5, 6, 7, 8, 9)
#    - The 'shape' argument defines the tensor size by its dimensions.
#    - 1st dimension = None. Indicates that the batch size, can be of any size.
#    - 2nd dimension = 10. Indicates the number of targets/outcomes

# dtype for both placeholders: if you not sure, use tf.float32. The limitation here is that the later presented softmax
# function only accepts float32 or float64 dtypes. For more dtypes, check TensorFlow's documentation
# https://www.tensorflow.org/api_docs/python/tf/DType

# ## Assign bias and weights to null tensors
# Now we are going to create the weights and biases, for this purpose they will be used as arrays filled with zeros.
# The values that we choose here can be critical, but we'll cover a better way on the second part, instead of this type
# of initialization.
# Weight tensor
W = tf.Variable(tf.zeros([784, 10], tf.float32))
# Bias tensor
b = tf.Variable(tf.zeros([10], tf.float32))

# ## Execute the assignment operation
# Before, we assigned the weights and biases but we did not initialize them with null values. For this reason,
# TensorFlow need to initialize the variables that you assign.
# Please notice that we're using this notation "sess.run" because we previously started an interactive session.
# Run the op initialize_all_variables using an interactive session
init = tf.compat.v1.global_variables_initializer()
sess.run(init)

# ## Adding Weights and Biases to input
# The tf.matmul operation performs a matrix multiplication between x (inputs) and W (weights). Then add biases.
z = tf.matmul(x, W) + b

# ## Softmax Regression
# Softmax is an activation function that is normally used in classification problems. It generate the probabilities for
# the output. For example, our model will not be 100% sure that one digit is the number nine, instead, the answer will
# be a distribution of probabilities where, if the model is right, the nine number will have a larger probability than
# the other other digits.
# For comparison, below is the one-hot vector for a nine digit label:
# 0 --> 0
# 1 --> 0
# 2 --> 0
# 3 --> 0
# 4 --> 0
# 5 --> 0
# 6 --> 0
# 7 --> 0
# 8 --> 0
# 9 --> 1
# A machine does not have all this certainty, so we want to know what is the best guess, but we also want to understand
# how sure it was and what was the second better option. Below is an example of a hypothetical distribution for a nine
# digit:
# 0 --> 0.01
# 1 --> 0.02
# 2 --> 0.03
# 3 --> 0.02
# 4 --> 0.12
# 5 --> 0.01
# 6 --> 0.03
# 7 --> 0.06
# 8 --> 0.1
# 9 --> 0.6
y_MLP = tf.nn.softmax(tf.matmul(x, W) + b)

# Logistic function output is used for the classification between two target classes 0/1. Softmax function is
# generalized type of logistic function. That is, Softmax can output a multiclass categorical probability distribution.

# ## Cost function
# It is a function that is used to minimize the difference between the right answers (labels) and estimated outputs by
# our Network.
cross_entropy = tf.reduce_mean(input_tensor=-tf.reduce_sum(input_tensor=y_ * tf.math.log(y_MLP), axis=[1]))

# ## Type of optimization: Gradient Descent
# This is the part where you configure the optimizer for your Neural Network. There are several optimizers available,
# in our case we will use Gradient Descent because it is a well established optimizer.
optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=0.5).minimize(cross_entropy)

# ## Train using minibatch Gradient Descent
# In practice, Batch Gradient Descent is not often used because is too computationally expensive. The good part about
# this method is that you have the true gradient, but with the expensive computing task of using the whole dataset in
# one time. Due to this problem, Neural Networks usually use minibatch to train.
# Load 50 training examples for each training iteration
print('\nStarting to train the model:')
for i in range(1000):
    batch = mnist.train.next_batch(batch_size)
    sess.run(optimizer, feed_dict={x: batch[0], y_: batch[1]})

# ## Evaluate the model
# Define prediction
correct_prediction = tf.equal(tf.argmax(input=y_MLP, axis=1), tf.argmax(input=y_, axis=1))
# Define accuracy
accuracy = tf.reduce_mean(input_tensor=tf.cast(correct_prediction, tf.float32))
# Evaluate accuracy in batches to avoid out-of-memory issues
acc = accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels}) * 100
# Print accuracy
print("\nThe final test accuracy for the simple ANN model is: {} %.".format(acc))

# ## End the session
sess.close()

# Is the final result good?
# Let's check the best algorithm available out there (10th June 2016):
# Result: 0.21% error (99.79% accuracy)  (ref. http://cs.nyu.edu/~wanli/dropc/)

# How to improve our model? Several options as follow:
#    - Regularization of Neural Networks using DropConnect
#    - Multi-column Deep Neural Networks for Image Classification
#    - APAC: Augmented Pattern Classification with Neural Networks
#    - Simple Deep Neural Network with Dropout
# In the next part we are going to explore the option:
#    - Simple Deep Neural Network with Dropout (more than 1 hidden layer)


# -----------------
# #### 2nd part: Deep Learning applied on MNIST.
print('\n\n ---> Deep Learning with CNN for MNIST classification.')
# In the first part, we learned how to use a simple ANN to classify MNIST. Now we are going to expand our knowledge
# using a Deep Neural Network. The architecture of such CNN is:
# - (Input) -> [batch_size, 28, 28, 1]  >> Apply 32 filter of [5x5]
# - (Convolutional layer 1)  -> [batch_size, 28, 28, 32]
# - (ReLU 1)  -> [?, 28, 28, 32]
# - (Max pooling 1) -> [?, 14, 14, 32]
# - (Convolutional layer 2)  -> [?, 14, 14, 64]
# - (ReLU 2)  -> [?, 14, 14, 64]
# - (Max pooling 2)  -> [?, 7, 7, 64]
# - [fully connected layer 3] -> [1x1024]
# - [ReLU 3]  -> [1x1024]
# - [Drop out]  -> [1x1024]
# - [fully connected layer 4] -> [1x10]


# *---* Build the network *---*

# ## Create an interactive session
# End possible remaining session
sess.close()
# Start interactive session
sess = tf.compat.v1.InteractiveSession()


# ## Initial parameters
# Create general parameters for the model
width = 28  # width of the image in pixels
height = 28  # height of the image in pixels
flat = width * height  # number of pixels in one image
class_output = 10  # number of possible classifications for the problem

# ## Input and output
# Create place holders for inputs and outputs
x = tf.compat.v1.placeholder(tf.float32, shape=[None, flat])
y_ = tf.compat.v1.placeholder(tf.float32, shape=[None, class_output])

# ## Converting images of the data set to tensors
# The input image is 28 pixels by 28 pixels, 1 channel (grayscale). In this case, the first dimension is the batch
# number of the image, and can be of any size (so we set it to -1). The second and third dimensions are width and
# height, and the last one is the image channels.
x_image = tf.reshape(x, [-1, 28, 28, 1])

# ### -->  1) Convolutional Layer 1

# ## Defining kernel weight and bias
# We define a kernel here. The Size of the filter/kernel is 5x5;  Input channels is 1 (grayscale);  and we need 32
# different feature maps (here, 32 feature maps means 32 different filters are applied on each image. So, the output
# of convolution layer would be 28x28x32). In this step, we create a filter / kernel tensor of shape:
# [filter_height, filter_width, in_channels, out_channels]
W_conv1 = tf.Variable(tf.random.truncated_normal([5, 5, 1, 32], stddev=0.1))
b_conv1 = tf.Variable(tf.constant(0.1, shape=[32]))  # need 32 biases for 32 outputs

# ## Convolve with weight tensor and add biases.
# To create convolutional layer, we use tf.nn.conv2d. It computes a 2-D convolution given 4-D input and filter tensors.
convolve1 = tf.nn.conv2d(input=x_image, filters=W_conv1, strides=[1, 1, 1, 1], padding='SAME') + b_conv1

# Inputs:
# - tensor of shape [batch, in_height, in_width, in_channels]. x of shape [batch_size,28 ,28, 1]
# - a filter/kernel tensor of shape [filter_height, filter_width, in_channels, out_channels]. W is of size [5, 5, 1, 32]
# - stride which is [1, 1, 1, 1]. The convolutional layer, slides the "kernel window" across the input tensor. As the
# input tensor has 4 dimensions: [batch, height, width, channels], then the convolution operates on a 2D window on the
# height and width dimensions. strides determines how much the window shifts by in each of the dimensions. As the first
# and last dimensions are related to batch and channels, we set the stride to 1. But for second and third dimension,
# we could set other values, e.g. [1, 2, 2, 1]
#
# Process:
# - Change the filter to a 2-D matrix with shape [5\*5\*1,32]
# - Extracts image patches from the input tensor to form a *virtual* tensor of shape [batch, 28, 28, 5*5*1].
# - For each batch, right-multiplies the filter matrix and the image vector.
#
# Output:
# - A `Tensor` (a 2-D convolution) of size tf.Tensor 'add_7:0' shape=(?, 28, 28, 32)- Notice: the output of the first
# convolution layer is 32 [28x28] images. Here 32 is considered as volume/depth of the output image.

# ## Apply the ReLU activation Function
# In this step, we just go through all outputs convolution layer, convolve1, and wherever a negative number occurs,
# we swap it out for a 0. It is called ReLU activation Function. Let f(x) be a ReLU activation function f(x) = max(0,x).
h_conv1 = tf.nn.relu(convolve1)

# ## Apply the max pooling
# Max pooling is a form of non-linear down-sampling. It partitions the input image into a set of rectangles and, and
# then find the maximum value for that region.
# Lets use tf.nn.max_pool function to perform max pooling.
#  - Kernel size: 2x2 (if the window is a 2x2 matrix, it would result in one output pixel)
#  - Strides: dictates the sliding behaviour of the kernel. In this case it will move 2 pixels everytime, thus not
#  overlapping. The input is a matrix of size 28x28x32, and the output would be a matrix of size 14x14x32.
conv1 = tf.nn.max_pool2d(input=h_conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')  # max_pool_2x2
print('\nThe first layer object, after applying convolution and max-pooling, is:')
print(conv1)
# The first layer is completed.

# ### -->  2) Convolutional Layer 2

# ## Weights and Biases of kernels
# We apply the convolution again in this layer. Lets look at the second layer kernel:
# - Filter/kernel: 5x5 (25 pixels)
# - Input channels: 32 (from the 1st Conv layer, we had 32 feature maps)
# - 64 output feature maps
# Notice: here, the input image is [14x14x32], the filter is [5x5x32], we use 64 filters of size [5x5x32], and the
# output of the convolutional layer would be 64 convolved image, [14x14x64].
# Notice: the convolution result of applying a filter of size [5x5x32] on image of size [14x14x32] is an image of
# size [14x14x1], that is, the convolution is functioning on volume.
W_conv2 = tf.Variable(tf.random.truncated_normal([5, 5, 32, 64], stddev=0.1))
b_conv2 = tf.Variable(tf.constant(0.1, shape=[64]))  # need 64 biases for 64 outputs

# ## Convolve image with weight tensor and add biases.
convolve2 = tf.nn.conv2d(input=conv1, filters=W_conv2, strides=[1, 1, 1, 1], padding='SAME') + b_conv2

# ## Apply the ReLU activation Function
h_conv2 = tf.nn.relu(convolve2)

# ## Apply the max pooling
conv2 = tf.nn.max_pool2d(input=h_conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')  # max_pool_2x2
print('The second layer object, after applying convolution and max-pooling, is:')
print(conv2)
# Second layer completed. So, what is the output of the second layer? It is 64 matrix of [7x7].

# ### -->  3) Fully Connected Layer

# You need a fully connected layer to use the Softmax and create the probabilities in the end. Fully connected layers
# take the high-level filtered images from previous layer, that is all 64 matrices, and convert them to a flat array.
# So, each matrix [7x7] will be converted to a matrix of [49x1], and then all of the 64 matrix will be connected, which
# make an array of size [3136x1]. We will connect it into another layer of size [1024x1]. So, the weight between these
# 2 layers will be [3136x1024].

# ## Flattening Second Layer
layer2_matrix = tf.reshape(conv2, [-1, 7 * 7 * 64])

# ## Weights and Biases between layer 2 and 3
# Composition of the feature map from the last layer (7x7) multiplied by the number of feature maps (64); 1027 outputs
# to Softmax layer
W_fc1 = tf.Variable(tf.random.truncated_normal([7 * 7 * 64, 1024], stddev=0.1))
b_fc1 = tf.Variable(tf.constant(0.1, shape=[1024]))  # need 1024 biases for 1024 outputs

# ## Matrix Multiplication (applying weights and biases)
fcl = tf.matmul(layer2_matrix, W_fc1) + b_fc1

# ## Apply the ReLU activation Function
h_fc1 = tf.nn.relu(fcl)
print('The third layer object, consisting of a fully-connected layer with ReLu activation, is:')
print(h_fc1)

# ##  Dropout: Optional phase for reducing overfitting
# It is a phase where the network "forget" some features. At each training step in a mini-batch, some units get switched
# off randomly so that it will not interact with the network. That is, its weights cannot be updated, nor affect the
# learning of the other network nodes. This can be very useful for very large neural networks to prevent overfitting.
keep_prob = tf.compat.v1.placeholder(tf.float32)
layer_drop = tf.nn.dropout(h_fc1, 1 - keep_prob)
print('After applying dropout on the third layer the corresponding object is:')
print(layer_drop)
# Third layer completed.

# ### -->  4) Readout Layer (softmax fully-connected layer)

# ## Weights and Biases
# In last layer, CNN takes the high-level filtered images and translate them into votes using softmax.
# Input channels: 1024 (neurons from the 3rd Layer); 10 output features
W_fc2 = tf.Variable(tf.random.truncated_normal([1024, 10], stddev=0.1))  # 1024 neurons
b_fc2 = tf.Variable(tf.constant(0.1, shape=[10]))  # 10 possibilities for digits [0,1,2,3,4,5,6,7,8,9]

# ## Matrix Multiplication (applying weights and biases)
fc = tf.matmul(layer_drop, W_fc2) + b_fc2

# ## Apply the Softmax activation Function
# softmax allows us to interpret the outputs of fcl4 as probabilities. So, y_conv is a tensor of probabilities.
y_CNN = tf.nn.softmax(fc)
print('The tensor of probabilities coming out of the fourth layer, i.e. the readout (fully-connected softmax), is:')
print(y_CNN)
# Last layer is completed.

# ## Summary of the Deep Convolutional Neural Network
# Now is time to remember the structure of our network
# #### 0) Input - MNIST dataset
# #### 1) Convolutional and Max-Pooling (layer 1)
# #### 2) Convolutional and Max-Pooling (layer 2)
# #### 3) Fully Connected Layer (layer 3)
# #### 4) Processing - Dropout
# #### 5) Readout layer - Fully Connected (layer 4)
# #### 6) Outputs - Classified digits


# *---* Define functions and train the model *---*

# ## Define the loss function
# We need to compare our output, layer4 tensor, with ground truth for all mini_batch. we can use cross entropy to see
# how bad our CNN is working - to measure the error at a softmax layer.
cross_entropy = tf.reduce_mean(input_tensor=-tf.reduce_sum(input_tensor=y_ * tf.math.log(y_CNN), axis=[1]))
# reduce_sum computes the sum of elements of (y_ * tf.log(layer4)) across second dimension of the tensor, and
# reduce_mean computes the mean of all elements in the tensor.

# ## Define the optimizer
# It is obvious that we want minimize the error of our network which is calculated by cross_entropy metric. To solve
# the problem, we have to compute gradients for the loss (which is minimizing the cross-entropy) and apply gradients to
# variables. It will be done by an optimizer: GradientDescent or Adagrad.
train_step = tf.compat.v1.train.AdamOptimizer(learning_rate=1e-4).minimize(cross_entropy)

# ## Define prediction
# Do you want to know how many of the cases in a mini-batch has been classified correctly? Lets count them.
correct_prediction = tf.equal(tf.argmax(input=y_CNN, axis=1), tf.argmax(input=y_, axis=1))

# ## Define accuracy
# It makes more sense to report accuracy using average of correct cases.
accuracy = tf.reduce_mean(input_tensor=tf.cast(correct_prediction, tf.float32))

# ## Initializing the variables
init = tf.compat.v1.global_variables_initializer()
sess.run(init)


# ## Train the network
# Define the function for training the model
def train(n_epochs):
    print('\nStarting to train the model:')

    for i in range(n_epochs):
        batch = mnist.train.next_batch(batch_size)

        if i % 100 == 0:
            train_accuracy = accuracy.eval(feed_dict={x: batch[0], y_: batch[1], keep_prob: 1.0})
            print("At step %d training accuracy is %g" % (i, float(train_accuracy)))

        train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})


# Train the model (it might take some time)
epochs = 1100  # 20000
# Note: if you REALLY have time to wait, or you are training the model using PowerAI, and want better results, change
# the number of epochs to 20000.
train(epochs)


# *---* Evaluate the model *---*

# ## Lets evaluate and print test accuracy
# Evaluate in batches to avoid out-of-memory issues
n_batches = mnist.test.images.shape[0] // batch_size
cumulative_accuracy = 0.0
for index in range(n_batches):
    batch = mnist.test.next_batch(batch_size)
    cumulative_accuracy += accuracy.eval(feed_dict={x: batch[0], y_: batch[1], keep_prob: 1.0})
acc = cumulative_accuracy / n_batches * 100
# Print the evaluation to the user.
print("\nThe final test accuracy for the CNN model is:  {} %.".format(round(acc, 2)))

# ## Visualization
# Do you want to look at all the filters?
kernels = sess.run(tf.reshape(tf.transpose(a=W_conv1, perm=[2, 3, 0, 1]), [32, -1]))
image = Image.fromarray(tile_raster_images(kernels, img_shape=(5, 5), tile_shape=(4, 8), tile_spacing=(1, 1)))

# Plot all the kernels in the conv_1 layer
plt.rcParams['figure.figsize'] = (18.0, 18.0)
plt.imshow(image, cmap='gray')
plt.title('Kernels of Conv_1 after learning')
plt.show()

# Lets see the output of an image passing through first convolution layer. First lets take a sample image:
plt.rcParams['figure.figsize'] = (5.0, 5.0)
sampleimage = mnist.test.images[1]
plt.imshow(np.reshape(sampleimage, [28, 28]), cmap="gray")
plt.title('Sample image')
plt.show()

# Resulting filters of Conv1:
ActivatedUnits = sess.run(convolve1, feed_dict={x: np.reshape(sampleimage, [1, 784], order='F'), keep_prob: 1.0})
filters = ActivatedUnits.shape[3]
plt.figure(1, figsize=(20, 20))
n_columns = 6
n_rows = np.math.ceil(filters / n_columns) + 1
for i in range(filters):
    plt.subplot(n_rows, n_columns, i + 1)
    plt.imshow(ActivatedUnits[0, :, :, i], interpolation="nearest", cmap="gray")
    plt.title('Filter ' + str(i))
plt.suptitle('Kernels of Conv_1 layer when sample image is the input to Conv_1')
plt.show()

# Resulting filters of Conv2:
ActivatedUnits = sess.run(convolve2, feed_dict={x: np.reshape(sampleimage, [1, 784], order='F'), keep_prob: 1.0})
filters = ActivatedUnits.shape[3]
plt.figure(1, figsize=(20, 20))
n_columns = 8
n_rows = np.math.ceil(filters / n_columns) + 1
for i in range(filters):
    plt.subplot(n_rows, n_columns, i + 1)
    plt.title('Filter ' + str(i) + ' of Conv_2 layer when\n sample image is the input to Conv_1')
    plt.imshow(ActivatedUnits[0, :, :, i], interpolation="nearest", cmap="gray")
plt.suptitle('Kernels of Conv_2 layer when sample image is the input to Conv_1')
plt.show()

# ## End the session
sess.close()
