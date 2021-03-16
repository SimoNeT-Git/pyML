#!/usr/bin/env python
# coding: utf-8

# In this lab, you will learn two important components in building a convolutional neural network. The first is applying
# an activation function, which is analogous to building a regular network. You will also learn about max pooling. Max
# pooling reduces the number of parameters and makes the network less susceptible to changes in the image.
#   - Activation Functions
#   - Max Pooling

import torch
import torch.nn as nn


# #### Activation Functions

# Just like a neural network, you apply an activation function to the activation map.
# Create a kernel and image as usual. Set the bias to zero:
print("\n\n\n--> Convolution:")
conv = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3)
Gx = torch.tensor([[1.0, 0, -1.0], [2.0, 0, -2.0], [1.0, 0, -1.0]])
conv.state_dict()['weight'][0][0] = Gx
conv.state_dict()['bias'][0] = 0.0

# Print information on convolutional object
print("\nSome information about the convolutional object:  ", conv)
print("Weights are:")
print(conv.state_dict()['weight'])
print("Bias is:")
print(conv.state_dict()['bias'])

# Create image
image = torch.zeros(1, 1, 5, 5)
image[0, 0, :, 2] = 1
print("\nThe simple custom-made black-white image is:")
print(image)

# Apply convolution to the image:
Z = conv(image)
print("\nAfter applying the convolutional kernel on the image we get:")
print(Z)
print("The shape is:", Z.shape[2:4])

# Apply the activation function to the activation map. This will apply the activation function to each element in the
# activation map.
A = torch.relu(Z)  # or: relu = nn.Relu(); A = relu(Z)
print("\nAfter applying the ReLu activation function on the convolutional output we get:")
print(A)
# The Relu function is applied to each element. All the elements less than zero are mapped to zero. The remaining
# components do not change.


# #### Max Pooling

print("\n\n\n--> Max Pooling:")
# Consider the following image:
image1 = torch.zeros(1, 1, 4, 4)
image1[0, 0, 0, :] = torch.tensor([1.0, 2.0, 3.0, -4.0])
image1[0, 0, 1, :] = torch.tensor([0.0, 2.0, -3.0, 0.0])
image1[0, 0, 2, :] = torch.tensor([0.0, 2.0, 3.0, 1.0])
print("\nThe custom-made image on which to apply max-pooling is:")
print(image1)

# Max pooling simply takes the maximum value in each region. Consider the following image. For the first region, max
# pooling simply takes the largest element in a yellow region.
# The region shifts, and the process is repeated. The process is similar to convolution.
# Create a maxpooling object in 2d as follows and perform max pooling as follows:
maxpool1 = torch.nn.MaxPool2d(2, stride=1)
print("\nSome information about the first max-pooling object:  ", maxpool1)
MP1 = maxpool1(image1)
print("\nAfter applying the first max-pooling object on the image we get:")
print(MP1)

# If the stride is set to None (its defaults setting), the process will simply take the maximum in a prescribed area and
# shift over accordingly.
maxpool2 = torch.nn.MaxPool2d(2)
print("\nSome information about the second max-pooling object:  ", maxpool2)
MP2 = maxpool2(image1)
print("\nAfter applying the second max-pooling object on the image we get:")
print(MP2)
