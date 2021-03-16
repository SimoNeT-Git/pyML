#!/usr/bin/env python
# coding: utf-8

# In this lab, you will study convolution and review how the different operations change the relationship between input
# and output.
#   - What is Convolution
#   - Determining the Size of Output
#   - Stride
#   - Zero Padding

import torch
import torch.nn as nn


# #### What is Convolution?

# Convolution is a linear operation similar to a linear equation, dot product, or matrix multiplication. Convolution has
# several advantages for analyzing images. Convolution preserves the relationship between elements, and it requires
# fewer parameters than other methods.
# In convolution, the parameter w is called a kernel. You can perform convolution on images where you let the variable
# image denote the variable X and w denote the parameter.

# Create a two-dimensional convolution object by using the constructor Conv2d, the parameter in_channels and
# out_channels will be used for this section, and the parameter kernel_size will be three.
print("\n\n\n--> Example 1: 3x3 kernel")
conv = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3)
print("\nSome information about the convolutional object:  ", conv)
print("Weights are:")
print(conv.state_dict()['weight'])
print("Bias is:")
print(conv.state_dict()['bias'])

# Because the parameters in nn.Conv2d are randomly initialized and learned through training, give them some values.
conv.state_dict()['weight'][0][0] = torch.tensor([[1.0, 0, -1.0], [2.0, 0, -2.0], [1.0, 0.0, -1.0]])
conv.state_dict()['bias'][0] = 0.0
print('After setting custom-made parameters we have:')
print("New weights:")
print(conv.state_dict()['weight'])
print("New bias:")
print(conv.state_dict()['bias'])

# Create a dummy tensor to represent an image. The shape of the image is:
# (number of inputs, number of channels, number of rows, number of columns) = (1,1,5,5)
# Set the third column to 1:
image = torch.zeros(1, 1, 5, 5)
image[0, 0, :, 2] = 1
print("\nThe simple custom-made black-white image is:")
print(image)

# Call the object conv on the tensor image as an input to perform the convolution and assign the result to the tensor z.
# The kernel performs at the element-level multiplication on every element in the image in the corresponding region.
# The values are then added together. The kernel is then shifted and the process is repeated.
z = conv(image)
print("\nAfter applying the convolutional kernel on the image we get:")
print(z)
print("The shape is:", z.shape[2:4])


# #### Determining the Size of the Output

# The size of the output is an important parameter. In this lab, you will assume square images. For rectangular images,
# the same formula can be used in for each dimension independently.
# Let M be the size of the input and K be the size of the kernel. The size of the output is given by the following
# formula: M_new = M - K + 1

# Create a kernel of size 2:
print("\n\n\n--> Example 2: 2x2 kernel")
K = 2
conv1 = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=K)
conv1.state_dict()['weight'][0][0] = torch.tensor([[1.0, 1.0], [1.0, 1.0]])
conv1.state_dict()['bias'][0] = 0.0
print("\nSome information about the convolutional object:  ", conv1)
print("Weights are:")
print(conv1.state_dict()['weight'])
print("Bias is:")
print(conv1.state_dict()['bias'])

# Create an image of size 2:
M = 4
image1 = torch.ones(1, 1, M, M)
print("\nThe simple custom-made black-white image is:")
print(image1)

# The following equation provides the output: M_new = M - K + 1 = 4 - 2 + 1 = 3
# The first iteration of the kernel overlay of the images produces one output. As the kernel is of size K, there are M-K
# elements for the kernel to move in the horizontal direction. The same logic applies to the vertical direction.

# Perform the convolution and verify the size is correct:
z1 = conv1(image1)
print("\nAfter applying the convolutional kernel on the image we get:")
print(z1)
print("The shape is:", z1.shape[2:4])


# #### Stride parameter

# The parameter stride changes the number of shifts the kernel moves per iteration. As a result, the output size also
# changes and is given by the following formula: M_new = (M-K) / stride + 1

# Create a convolution object with a stride of 2:
print("\n\n\n--> Example 3: stride=2")
conv2 = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=2, stride=2)
conv2.state_dict()['weight'][0][0] = torch.tensor([[1.0, 1.0], [1.0, 1.0]])
conv2.state_dict()['bias'][0] = 0.0
print("\nSome information about the convolutional object:  ", conv2)
print("Weights are:")
print(conv2.state_dict()['weight'])
print("Bias is:")
print(conv2.state_dict()['bias'])

# For an image with a size of 4, calculate the output size: M_new = (M-K) / stride + 1 = (4-2) / 2 + 1 = 2
# The first iteration of the kernel overlay of the images produces one output. Because the kernel is of size K, there
# are M-K=2 elements. The stride is 2 because it will move 2 elements at a time. As a result, you divide M-K by the
# stride value 2.

# The image
print("\nThe simple custom-made black-white image is:")
print(image1)

# Perform the convolution and verify the size is correct:
z2 = conv2(image1)
print("\nAfter applying the convolutional kernel on the image we get:")
print(z2)
print("The shape is:", z2.shape[2:4])


# #### Zero Padding

# As you apply successive convolutions, the image will shrink. You can apply zero padding to keep the image at
# reasonable size, which also holds information at the borders.
# In addition, you might not get integer values for the size of the kernel. Consider image1.
# Try performing convolutions with the kernel_size=2 and a stride=3. Use these values: M_new = (M-K) / stride + 1 =
# (4-2) / 3 + 1 = 1.666
print("\n\n\n--> Example 4: stride=3")
conv3 = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=2, stride=3)
conv3.state_dict()['weight'][0][0] = torch.tensor([[1.0, 1.0], [1.0, 1.0]])
conv3.state_dict()['bias'][0] = 0.0
print("\nSome information about the convolutional object:  ", conv3)
print("Weights are:")
print(conv3.state_dict()['weight'])
print("Bias is:")
print(conv3.state_dict()['bias'])
print("\nThe simple custom-made black-white image is:")
print(image1)
z3 = conv3(image1)
print("\nAfter applying the convolutional kernel on the image we get:")
print(z3)
print("The shape is:", z3.shape[2:4])

# You can add rows and columns of zeros around the image. This is called padding. In the constructor Conv2d, you specify
# the number of rows or columns of zeros that you want to add with the parameter padding.
# For a square image, you merely pad an extra column of zeros to the first column and the last column. Repeat the
# process for the rows. As a result, for a square image, the width and height is the original size plus 2 times the
# number of padding elements specified. You can then determine the size of the output after subsequent operations
# accordingly as shown in the following equation where you determine the size of an image after padding and then
# applying a convolutions kernel of size K: M'= M + 2 * padding, M_new = M' - K + 1

# Consider the following example:
print("\n\n\n--> Example 5: zero-padding=1")
conv4 = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=2, stride=3, padding=2)
conv4.state_dict()['weight'][0][0] = torch.tensor([[1.0, 1.0], [1.0, 1.0]])
conv4.state_dict()['bias'][0] = 0.0
print("\nSome information about the convolutional object:  ", conv4)
print("Weights are:")
print(conv4.state_dict()['weight'])
print("Bias is:")
print(conv4.state_dict()['bias'])
print("\nThe simple custom-made black-white image is:")
print(image1)
z4 = conv4(image1)
print("\nAfter applying the convolutional kernel and zero-padding on the image we get:")
print(z4)
print("The shape is:", z4.shape[2:4])


## Question 1:
# A kernel of zeros with a kernel size=3 is applied to the following image:
image2 = torch.randn((1, 1, 4, 4))
# Without using the function, determine what the outputs values are as each element:
# Answer:
# As each element of the kernel is zero, and for every output, the image is multiplied by the kernel, the result is
# always zero.

## Question 2:
# Use the following convolution object to perform convolution on the tensor Image:
print("\n\n\n--> Example 6: random image")
conv5 = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3)
conv5.state_dict()['weight'][0][0] = torch.tensor([[0, 0, 0], [0, 0, 0], [0, 0.0, 0]])
conv5.state_dict()['bias'][0] = 0.0
print("\nSome information about the convolutional object:  ", conv5)
print("Weights are:")
print(conv5.state_dict()['weight'])
print("Bias is:")
print(conv5.state_dict()['bias'])
print("\nThe randomly generated simple image is:")
print(image2)
# Answer:
z5 = conv5(image2)
print("\nAfter applying the convolutional kernel on the image we get:")
print(z5)
print("The shape is:", z5.shape[2:4])

## Question 3:
# You have an image of size 4. The parameters are as follows kernel_size=2,stride=2. What is the size of the output?
# Answer:
# (M-K)/stride + 1 = (4-2)/2 + 1 = 2
