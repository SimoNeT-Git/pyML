#!/usr/bin/env python
# coding: utf-8

# In this lab, you will use a prebuilt dataset and then use some prebuilt dataset transforms.
#  - Prebuilt Datasets
#  - Torchvision Transforms

import torch
import matplotlib.pylab as plt
import numpy as np
import torchvision.transforms as transforms
import torchvision.datasets as dsets

torch.manual_seed(0)


# This is the function for displaying images: show data by diagram
def show_data(data_sample, shape=(28, 28)):
    plt.imshow(data_sample[0].numpy().reshape(shape), cmap='gray')
    plt.title('y = ' + str(data_sample[1]))


# We can import a prebuilt dataset. In this case, use MNIST. You'll work with several of these parameters later by
# placing a transform object in the argument transform.

# Import the prebuilt dataset into variable dataset
dataset = dsets.MNIST(
    root='./data',
    train=False,
    download=True,
    transform=transforms.ToTensor()
)
# Each element of the dataset object contains a tuple. Let us see whether the first element in the dataset is a tuple
# and what is in it.

# Examine whether the elements in dataset MNIST are tuples, and what is in the tuple?
print("Info for MNIST dataset.")
print("Type of the first element: ", type(dataset[0]))
print("The length of the tuple: ", len(dataset[0]))
print("The shape of the first element in the tuple: ", dataset[0][0].shape)
print("The type of the first element in the tuple", type(dataset[0][0]))
print("The second element in the tuple: ", dataset[0][1])
print("The type of the second element in the tuple: ", type(dataset[0][1]))
print("As the result, the structure of the first element in the dataset is (tensor([1, 28, 28]), tensor(7)).")
# As shown in the output, the first element in the tuple is a cuboid tensor. As you can see, there is a dimension with
# only size 1, so basically, it is a rectangular tensor.
# The second element in the tuple is a number tensor, which indicate the real number the image shows. As the second
# element in the tuple is tensor(7), the image should show a hand-written 7.

# Let us plot the first element in the dataset:
plt.figure()
show_data(dataset[0])
plt.show()
# As we can see, it is a 7.

# Plot the second element in the dataset (the second sample)
plt.figure()
show_data(dataset[1])
plt.show()
# It is a 2.

# We can apply some image transform functions on the MNIST dataset.

# As an example, the images in the MNIST dataset can be cropped and converted to a tensor. We can use transform.Compose
# we learned from the previous lab to combine the two transform functions.
dim = 20
croptensor_data_transform = transforms.Compose([transforms.CenterCrop(dim), transforms.ToTensor()])
dataset = dsets.MNIST(
    root='./data',
    train=False,
    download=True,
    transform=croptensor_data_transform  # Apply crop and convert to tensor
)
print("\nAfter transformation, the shape of the first element in the first tuple is: ", dataset[0][0].shape)
# We can see the image is now 20 x 20 instead of 28 x 28.

# Let us plot the first image again. Notice that the black space around the 7 become less apparent.
plt.figure()
show_data(dataset[0], shape=(dim, dim))
plt.show()

# Plot the second element in the dataset
plt.figure()
show_data(dataset[1], shape=(dim, dim))
plt.show()

# In the below example, we horizontally flip the image, and then convert it to a tensor. Use transforms.Compose()
# to combine these two transform functions. Plot the flipped image.

# Construct the compose. Apply it on MNIST dataset. Plot the image out.
fliptensor_data_transform = transforms.Compose([transforms.RandomHorizontalFlip(p=1), transforms.ToTensor()])
dataset = dsets.MNIST(
    root='./data',
    train=False,
    download=True,
    transform=fliptensor_data_transform  # Apply horizontal flip and convert to tensor
)
plt.figure()
show_data(dataset[0])
plt.show()

# Combine vertical flip, horizontal flip and convert to tensor as a compose.
my_data_transform = transforms.Compose(
    [transforms.RandomVerticalFlip(p=1), transforms.RandomHorizontalFlip(p=1), transforms.ToTensor()])
dataset = dsets.MNIST(
    root='./data',
    train=False,
    download=True,
    transform=my_data_transform
)
plt.figure()
show_data(dataset[0])
plt.show()
