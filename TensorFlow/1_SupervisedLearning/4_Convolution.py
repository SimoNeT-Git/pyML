#!/usr/bin/env python
# coding: utf-8

# In this lesson, we will learn more about the key concepts behind the CNNs (Convolutional Neural Networks from now on).
# This lesson is not intended to be a reference for machine learning, deep learning, convolutions or TensorFlow.
# The intention is to give notions to the user about these fields.
#   - Analogies
#   - Understanding and coding with Python
#   - Coding with TensorFlow
#   - Convolution applied on images

import tensorflow as tf
import numpy as np
from scipy import signal as sg
from scipy import misc
import matplotlib.pyplot as plt
from PIL import Image

# -----------------
# #### Analogies

# There are several ways to understand Convolutional Layers without using a mathematical approach. We are going to
# explore some of the ideas proposed by the Machine Learning community.
#
# Instances of Neurons
#
# When you start to learn a programming language, one of the first phases of your development is the learning and
# application of functions. Instead of rewriting pieces of code everytime that you would, a good student is encouraged
# to code using functional programming, keeping the code organized, clear and concise.
# CNNs can be thought of as a simplification of what is really going on, a special kind of neural network which uses
# identical copies of the same neuron. These copies include the same parameters (shared weights and biases) and
# activation functions.
#
# Location and type of connections
#
# In a fully connected layer NN, each neuron in the current layer is connected to every neuron in the previous layer,
# and each connection has it's own weight. This is a general purpose connection pattern and makes no assumptions about
# the features in the input data thus not taking any advantage that the knowledge of the data being used can bring.
# These types of layers are also very expensive in terms of memory and computation.
#
# In contrast, in a convolutional layer each neuron is only connected to a few nearby local neurons in the previous
# layer, and the same set of weights is used to connect to them.
#
# Feature Learning
#
# Feature engineering is the process of extracting useful patterns from input data that will help the prediction model
# to understand better the real nature of the problem. A good feature learning will present patterns in a way that
# significantly increase the accuracy and performance of the applied machine learning algorithms in a way that would
# otherwise be impossible or too expensive by just machine learning itself.
#
# Feature learning algorithms finds the common patterns that are important to distinguish between the wanted classes and
# extract them automatically. After this process, they are ready to be used in a classification or regression problem.
#
# The great advantage of CNNs is that they are uncommonly good at finding features in images that grow after each level,
# resulting in high-level features in the end. The final layers (can be one or more) use all these generated features
# for classification or regression.
#
# Basically, Convolutional Neural Networks is your best friend to automatically do Feature Engineering (Feature
# Learning) without wasting too much time creating your own codes and with no prior need of expertise in the field of
# Feature Engineering.

# -----------------
# #### Understanding and coding with Python

# Convolution: 1D operation with Python (Numpy/Scipy)

# Mathematical notation
# In this first example, we will use the pure mathematical notation. Here we have a one dimensional convolution
# operation. Lets say h is our image and x is our kernel:
#
# x[i] = { 3, 4, 5 }
# h[i] = { 2, 1, 0 }
#
# where i = index
#
# To use the convolution operation between the two arrays try the code below to see how easy it is to do in Python.

h = [2, 1, 0]
x = [3, 4, 5]
y = np.convolve(x, h)

# -----------------
# #### Analogies

# There are several ways to understand Convolutional Layers without using a mathematical approach. We are going to
# explore some of the ideas proposed by the Machine Learning community.
#
# Instances of Neurons
#
# When you start to learn a programming language, one of the first phases of your development is the learning and
# application of functions. Instead of rewriting pieces of code everytime that you would, a good student is encouraged
# to code using functional programming, keeping the code organized, clear and concise.
# CNNs can be thought of as a simplification of what is really going on, a special kind of neural network which uses
# identical copies of the same neuron. These copies include the same parameters (shared weights and biases) and
# activation functions.
#
# Location and type of connections
#
# In a fully connected layer NN, each neuron in the current layer is connected to every neuron in the previous layer,
# and each connection has it's own weight. This is a general purpose connection pattern and makes no assumptions about
# the features in the input data thus not taking any advantage that the knowledge of the data being used can bring.
# These types of layers are also very expensive in terms of memory and computation.
#
# In contrast, in a convolutional layer each neuron is only connected to a few nearby local neurons in the previous
# layer, and the same set of weights is used to connect to them.
#
# Feature Learning
#
# Feature engineering is the process of extracting useful patterns from input data that will help the prediction model
# to understand better the real nature of the problem. A good feature learning will present patterns in a way that
# significantly increase the accuracy and performance of the applied machine learning algorithms in a way that would
# otherwise be impossible or too expensive by just machine learning itself.
#
# Feature learning algorithms finds the common patterns that are important to distinguish between the wanted classes and
# extract them automatically. After this process, they are ready to be used in a classification or regression problem.
#
# The great advantage of CNNs is that they are uncommonly good at finding features in images that grow after each level,
# resulting in high-level features in the end. The final layers (can be one or more) use all these generated features
# for classification or regression.
#
# Basically, Convolutional Neural Networks is your best friend to automatically do Feature Engineering (Feature
# Learning) without wasting too much time creating your own codes and with no prior need of expertise in the field of
# Feature Engineering.

# -----------------
# #### Understanding and coding with Python

# Convolution: 1D operation with Python (Numpy/Scipy)

# Mathematical notation
# In this first example, we will use the pure mathematical notation. Here we have a one dimensional convolution
# operation. Lets say h is our image and x is our kernel:
#
# x[i] = { 3, 4, 5 }
# h[i] = { 2, 1, 0 }
#
# where i = index
#
# To use the convolution operation between the two arrays try the code below to see how easy it is to do in Python.

h = [2, 1, 0]
x = [3, 4, 5]
y = np.convolve(x, h)

# Now we are going to verify what Python did, because we don't trust computer outputs while we are learning. Using the
# equation of convolution for y[n]:
#
# $$y[n] = \sum\limits_{k\to-\infty}^\infty x[k] \cdot h[n-k] $$

print("Compare with the following values from Python: y[0] = {0} ; y[1] = {1}; y[2] = {2}; y[3] = {3}; y[4] = {4}"
      .format(y[0], y[1], y[2], y[3], y[4]))

# There are three methods to apply kernel on the matrix, with padding (full), with padding (same) and ]
# without padding (valid):
#
# Visually understanding the operation with padding (full)
#
# Lets think of the kernel as a sliding window. We have to come with the solution of padding zeros on the input array.
# This is a very famous implementation and will be easier to show how it works with a simple example, consider the case:
#
# x[i] = [6,2]
# h[i] = [1,2,5,4]
#
# Using the zero padding, we can calculate the convolution.
#
# You have to invert the filter x, otherwise the operation would be cross-correlation.
# First step, (now with zero padding):
x = [6, 2]
h = [1, 2, 5, 4]
y = np.convolve(x, h, "full")  # now, because of the zero padding, the final dimension of the array is bigger

# Visually understanding the operation with "same"
#
# In this approach, we just add the zero to left (and top of the matrix in 2D). That is, only the first 4 steps of
# "full" method:
x = [6, 2]
h = [1, 2, 5, 4]
y = np.convolve(x, h,
                "same")  # it is same as zero padding, but with returns an ouput with the same length as max of x or h

# Visually understanding the operation with no padding (valid)
#
# In the last case we only applied the kernel when we had a compatible position on the h array, in some cases you want
# a dimensionality reduction. For this purpose, we simple ignore the steps that would need padding:
#
# x[i] = [6 2]
#
# h[i] = [1 2 5 4]
#
# You have to invert the filter x, otherwise the operation would be cross-correlation.

# Let's verify with numpy
x = [6, 2]
h = [1, 2, 5, 4]
y = np.convolve(x, h, "valid")
# valid returns output of length max(x, h) - min(x, h) + 1, this is to ensure that values outside of the boundary of
# h will not be used in the calculation of the convolution in the next example we will understand why we used the
# argument valid

# Convolution: 2D operation with Python (Numpy/Scipy)
#
# The 2D convolution operation is defined as:
#
# $$ I'= \sum\limits_{u,v} I(x-u,y-v)g(u,v) $$
#
# Below we will apply the equation to an image represented by a 3x3 matrix according to the function g = (-1 1).
# Please note that when we apply the kernel we always use its inversion.
#
# We don't have to finish the calculations, we have the computer at our side. So, let's see what is the code to proceede
# with this operation:
I = [[255, 7, 3],
     [212, 240, 4],
     [218, 216, 230], ]
g = [[-1, 1]]
print('Without zero padding \n')
print('{0} \n'.format(sg.convolve(I, g, 'valid')))
# The 'valid' argument states that the output consists only of those elements
# that do not rely on the zero-padding.
print('With zero padding \n')
print(sg.convolve(I, g))
I = [[255, 7, 3],
     [212, 240, 4],
     [218, 216, 230], ]
g = [[-1, 1],
     [2, 3], ]
print('With zero padding \n')
print('{0} \n'.format(sg.convolve(I, g, 'full')))
# The output is the full discrete linear convolution of the inputs.
# It will use zero to complete the input matrix
print('With zero padding_same_ \n')
print('{0} \n'.format(sg.convolve(I, g, 'same')))
# The output is the full discrete linear convolution of the inputs.
# It will use zero to complete the input matrix
print('Without zero padding \n')
print(sg.convolve(I, g, 'valid'))
# The 'valid' argument states that the output consists only of those elements
# that do not rely on the zero-padding.

# -----------------
# #### Coding with TensorFlow
#
# Numpy is great because it has high optimized matrix operations implemented in a backend using C/C++. However, if our
# goal is to work with DeepLearning, we need much more. TensorFlow does the same work, but instead of returning to
# Python everytime, it creates all the operations in form of graphs and execute them once with highly optimized backend.

# Suppose that you have two tensors:
#   - 3x3 filter (4D tensor = [3,3,1,1] = [width, height, channels, number of filters])
#   - 10x10 image (4D tensor = [1,10,10,1] = [batch size, width, height, number of channels]
# The output size for zero padding 'SAME' mode will be the same as input = 10x10
# The output size without zero padding 'VALID' mode will be input size - kernel dimension + 1 = 10 -3 + 1 = 8 = 8x8

# Building graph
input = tf.Variable(tf.random.normal([1, 10, 10, 1]))
filter = tf.Variable(tf.random.normal([3, 3, 1, 1]))
op = tf.nn.conv2d(input=input, filters=filter, strides=[1, 1, 1, 1], padding='VALID')
op2 = tf.nn.conv2d(input=input, filters=filter, strides=[1, 1, 1, 1], padding='SAME')

# Initialization and session
init = tf.compat.v1.global_variables_initializer()
with tf.compat.v1.Session() as sess:
    sess.run(init)
    print("Input \n")
    print('{0} \n'.format(input.eval()))
    print("Filter/Kernel \n")
    print('{0} \n'.format(filter.eval()))
    print("Result/Feature Map with valid positions \n")
    result = sess.run(op)
    print(result)
    print('\n')
    print("Result/Feature Map with padding \n")
    result2 = sess.run(op2)
    print(result2)

# -----------------
# #### Convolution applied on images

# You can try convolution on a default image (bird.jpg). The result of this pre-processing will be an image with only a
# grayscale channel.
im = Image.open('bird.jpg')  # type here your image's name
image_gr = im.convert("L")  # convert("L") translate color images into black and white
# uses the ITU-R 601-2 Luma transform (there are several
# ways to convert an image to grey scale)
print("\n Original type: %r \n\n" % image_gr)

# convert image to a matrix with values from 0 to 255 (uint8) 
arr = np.asarray(image_gr)
print("After conversion to numerical representation: \n\n %r" % arr)

# Plot image
imgplot = plt.imshow(arr)
imgplot.set_cmap('gray')  # you can experiment different colormaps (Greys,winter,autumn)
print("\n Input image converted to gray scale: \n")
plt.show(imgplot)

# Now, we will experiment using an edge detector kernel.
kernel = np.array([[0, 1, 0],
                   [1, -4, 1],
                   [0, 1, 0], ])
grad = sg.convolve2d(arr, kernel, mode='same', boundary='symm')
print('GRADIENT MAGNITUDE - Feature map')

fig, aux = plt.subplots(figsize=(10, 10))
aux.imshow(np.absolute(grad), cmap='gray')

# If we change the kernel and start to analyze the outputs we would be acting as a CNN. The difference is that a Neural
# Network do all this work automatically (the kernel adjustment using different weights). In addition, we can understand
# how biases affect the behaviour of feature maps
# Please note that when you are dealing with most of the real applications of CNNs, you usually convert the pixels
# values to a range from 0 to 1. This process is called normalization.
type(grad)

grad_biases = np.absolute(grad) + 100

grad_biases[grad_biases > 255] = 255

print('GRADIENT MAGNITUDE - Feature map')

fig, aux = plt.subplots(figsize=(10, 10))
aux.imshow(np.absolute(grad_biases), cmap='gray')

# Lets see how it works for a digit:
im = Image.open('num3.jpg')  # type here your image's name
image_gr = im.convert("L")  # convert("L") translate color images into black and white
# uses the ITU-R 601-2 Luma transform (there are several
# ways to convert an image to grey scale)
print("\n Original type: %r \n\n" % image_gr)

# convert image to a matrix with values from 0 to 255 (uint8) 
arr = np.asarray(image_gr)
print("After conversion to numerical representation: \n\n %r" % arr)

# Plot image
fig, aux = plt.subplots(figsize=(10, 10))
imgplot = plt.imshow(arr)
imgplot.set_cmap('gray')  # you can experiment different colormaps (Greys,winter,autumn)
print("\n Input image converted to gray scale: \n")
plt.show(imgplot)

# Now, we will experiment using an edge detector kernel.
kernel = np.array([
    [0, 1, 0],
    [1, -4, 1],
    [0, 1, 0],
])

grad = sg.convolve2d(arr, kernel, mode='same', boundary='symm')

print('GRADIENT MAGNITUDE - Feature map')

fig, aux = plt.subplots(figsize=(10, 10))
aux.imshow(np.absolute(grad), cmap='gray')

# This understanding of how convolutions work are the foundation of how Convolutional Neural Networks work.
# After this tutorial you are supposed to understand the underlying mathematical concepts and how to apply them using
# Python (Numpy) and TensorFlow. The next step is to extrapolate this knowledge to Machine Learning applications.
