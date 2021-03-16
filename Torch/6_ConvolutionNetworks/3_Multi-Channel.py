#!/usr/bin/env python
# coding: utf-8

# In this lab, you will study convolution and review how the different operations change the relationship between input
# and output.
#   - Multiple Output Channels
#   - Multiple Inputs
#   - Multiple Input and Multiple Output Channels

import torch
import torch.nn as nn
import matplotlib.pyplot as plt


# #### Multiple Output Channels

print("\n\n\n--> Multiple Output Channels:")

# In Pytorch, you can create a Conv2d object with multiple outputs. For each channel, a kernel is created, and each
# kernel performs a convolution independently. As a result, the number of outputs is equal to the number of channels.
# Create a Conv2d with three channels:
conv1 = nn.Conv2d(in_channels=1, out_channels=3, kernel_size=3)
print("\nSome information about the convolutional object:  ", conv1)

# Pytorch randomly assigns values to each kernel. However, use kernels that have been developed to detect edges:
Gx = torch.tensor([[1.0, 0, -1.0], [2.0, 0, -2.0], [1.0, 0.0, -1.0]])
Gy = torch.tensor([[1.0, 2.0, 1.0], [0.0, 0.0, 0.0], [-1.0, -2.0, -1.0]])
Gz = torch.ones(3, 3)
conv1.state_dict()['weight'][0][0] = Gx
conv1.state_dict()['weight'][1][0] = Gy
conv1.state_dict()['weight'][2][0] = Gz
# Each kernel has its own bias, so set them all to zero:
conv1.state_dict()['bias'][:] = torch.tensor([0.0, 0.0, 0.0])

# Print kernel info
print("Weights are:")
print(conv1.state_dict()['weight'])
print("Bias is:")
print(conv1.state_dict()['bias'])

# Create an input image to represent the input X:
image1 = torch.zeros(1, 1, 5, 5)
image1[0, 0, :, 2] = 1
print("\nThe simple custom-made black-white image is:")
print(image1)

# Plot it as an image:
plt.figure()
plt.imshow(image1[0, 0, :, :].numpy(), interpolation='nearest', cmap=plt.cm.gray)
plt.title('Input Image')
plt.colorbar()
plt.show()

# Perform convolution using each channel: 
out1 = conv1(image1)
print("\nAfter applying the convolutional kernel on the image we get:")
print(out1)
# The result is a 1x3x3x3 tensor. This represents one sample with three channels, and each channel contains a 3x3 image.
# The same rules that govern the shape of each image were discussed in the last section.
print("The shape is:", out1.shape)

# Print out each channel as a tensor or an image:
fig = plt.figure(figsize=(16, 4))
for channel, image in enumerate(out1[0]):
    plt.subplot(1, len(out1[0]), channel + 1)
    plt.imshow(image.detach().numpy(), interpolation='nearest', cmap=plt.cm.gray)
    plt.title("channel {}".format(channel))
    plt.colorbar()
fig.suptitle('Multi-channel convolutional output')
plt.show()

# Different kernels can be used to detect various features in an image. You can see that the first channel fluctuates,
# and the second two channels produce a constant value.
# If you use a different image, the result will be different:
image2 = torch.zeros(1, 1, 5, 5)
image2[0, 0, 2, :] = 1
print("\nLets use another image:")
print(image2)
plt.figure()
plt.imshow(image2[0, 0, :, :].detach().numpy(), interpolation='nearest', cmap=plt.cm.gray)
plt.title('Input Image')
plt.show()

# In this case, the second channel fluctuates, and the first and the third channels produce a constant value.
out2 = conv1(image2)
print("\nThe convolutional output will now be:")
print(out2)
print("The shape is:", out2.shape)
fig = plt.figure(figsize=(16, 4))
for channel, image in enumerate(out2[0]):
    plt.subplot(1, len(out2[0]), channel + 1)
    plt.imshow(image.detach().numpy(), interpolation='nearest', cmap=plt.cm.gray)
    plt.title("channel {}".format(channel))
    plt.colorbar()
fig.suptitle('Multi-channel convolutional output')
plt.show()


# #### Multiple Input Channels

print("\n\n\n--> Multiple Input Channels:")

# Create a Conv2d object with two inputs:
conv2 = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=3)
print("\nSome information about the convolutional object:  ", conv2)

# Assign kernel values to make the math a little easier:
Gx1 = torch.tensor([[0.0, 0.0, 0.0], [0, 1.0, 0], [0.0, 0.0, 0.0]])
conv2.state_dict()['weight'][0][0] = 1 * Gx1
conv2.state_dict()['weight'][0][1] = -2 * Gx1
conv2.state_dict()['bias'][:] = torch.tensor([0.0])

# Print kernel info
print("Weights are:")
print(conv2.state_dict()['weight'])
print("Bias is:")
print(conv2.state_dict()['bias'])

# For two inputs, you can create two kernels. Each kernel performs a convolution on its associated input channel.
# Create an input with two channels:
image3 = torch.zeros(1, 2, 5, 5)
image3[0, 0, 2, :] = -2
image3[0, 1, 2, :] = 1
print("\nThe multi-channel input image is:")
print(image3)

# Plot out each image:
fig = plt.figure(figsize=(10, 4))
for channel, image in enumerate(image3[0]):
    plt.subplot(1, len(image3[0]), channel + 1)
    plt.imshow(image.detach().numpy(), interpolation='nearest', cmap=plt.cm.gray)
    plt.title("channel {}".format(channel))
    plt.colorbar()
fig.suptitle('Multi-channel input image')
plt.show()

# Perform the convolution:
out3 = conv2(image3)
print("\nThe convolutional output will now be:")
print(out3)
print("The shape is:", out3.shape)

# Plot the output
plt.figure()
plt.imshow(out3[0][0].detach().numpy(), interpolation='nearest', cmap=plt.cm.gray)
plt.title("Output of Multi-channel input")
plt.colorbar()
plt.show()


# #### Multiple Input and Multiple Output Channels

print("\n\n\n--> Multiple Input and Multiple Output Channels:")

# When using multiple inputs and outputs, a kernel is created for each input, and the process is repeated for each
# output.
# There are two input channels and 3 output channels. For each channel, the input in red and purple is convolved with an
# individual kernel that is colored differently. As a result, there are three outputs.
# Create an example with two inputs and three outputs:
conv3 = nn.Conv2d(in_channels=2, out_channels=3, kernel_size=3)
print("\nSome information about the convolutional object:  ", conv3)

# Assign the kernel values to make the math a little easier:
conv3.state_dict()['weight'][0][0] = torch.tensor([[0.0, 0.0, 0.0], [0, 0.5, 0], [0.0, 0.0, 0.0]])
conv3.state_dict()['weight'][0][1] = torch.tensor([[0.0, 0.0, 0.0], [0, 0.5, 0], [0.0, 0.0, 0.0]])

conv3.state_dict()['weight'][1][0] = torch.tensor([[0.0, 0.0, 0.0], [0, 1, 0], [0.0, 0.0, 0.0]])
conv3.state_dict()['weight'][1][1] = torch.tensor([[0.0, 0.0, 0.0], [0, -1, 0], [0.0, 0.0, 0.0]])

conv3.state_dict()['weight'][2][0] = torch.tensor([[1.0, 0, -1.0], [2.0, 0, -2.0], [1.0, 0.0, -1.0]])
conv3.state_dict()['weight'][2][1] = torch.tensor([[1.0, 2.0, 1.0], [0.0, 0.0, 0.0], [-1.0, -2.0, -1.0]])

# For each output, there is a bias, so set them all to zero:
conv3.state_dict()['bias'][:] = torch.tensor([0.0, 0.0, 0.0])

# Print kernel info
print("Weights are:")
print(conv3.state_dict()['weight'])
print("Bias is:")
print(conv3.state_dict()['bias'])

# Create a two-channel image and plot the results:
image4 = torch.zeros(1, 2, 5, 5)
image4[0][0] = torch.ones(5, 5)
image4[0][1][2][2] = 1
print("\nThe multi-channel input image is:")
print(image4)

# Plot out each input
fig = plt.figure(figsize=(10, 4))
for channel, image in enumerate(image4[0]):
    plt.subplot(1, len(image4[0]), channel + 1)
    plt.imshow(image.detach().numpy(), interpolation='nearest', cmap=plt.cm.gray)
    plt.title("channel {}".format(channel))
    plt.colorbar()
fig.suptitle('Multi-channel input image')
plt.show()

# Perform the convolution:
out4 = conv3(image4)
print("\nThe convolutional output will now be:")
print(out4)
print("The shape is:", out4.shape)
fig = plt.figure(figsize=(16, 4))
for channel, image in enumerate(out4[0]):
    plt.subplot(1, len(out4[0]), channel + 1)
    plt.imshow(image.detach().numpy(), interpolation='nearest', cmap=plt.cm.gray)
    plt.title("channel {}".format(channel))
    plt.colorbar()
fig.suptitle('Multi-channel convolutional output')
plt.show()

## Question:
# Use the following two convolution objects to produce the same result as two input channel convolution on imageA and
# imageB:
imageA = torch.zeros(1, 1, 5, 5)
imageB = torch.zeros(1, 1, 5, 5)
imageA[0, 0, 2, :] = -2
imageB[0, 0, 2, :] = 1

# Plot the inputs
# Plot out each image:
fig = plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.imshow(imageA[0][0].detach().numpy(), interpolation='nearest', cmap=plt.cm.gray)
plt.title("Image A")
plt.subplot(1, 2, 2)
plt.imshow(imageB[0][0].detach().numpy(), interpolation='nearest', cmap=plt.cm.gray)
plt.title("Image B")
plt.colorbar()
plt.show()

conv5 = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3)
conv6 = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3)
Gx1 = torch.tensor([[0.0, 0.0, 0.0], [0, 1.0, 0], [0.0, 0.0, 0.0]])
conv5.state_dict()['weight'][0][0] = 1 * Gx1
conv6.state_dict()['weight'][0][0] = -2 * Gx1
conv5.state_dict()['bias'][:] = torch.tensor([0.0])
conv6.state_dict()['bias'][:] = torch.tensor([0.0])

# Answer:
out5 = conv5(imageA)
out6 = conv6(imageB)
out = out5 + out6

# Plot out each image:
fig = plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.imshow(out5[0][0].detach().numpy(), interpolation='nearest', cmap=plt.cm.gray)
plt.title("Output A")
plt.subplot(1, 2, 2)
plt.imshow(out6[0][0].detach().numpy(), interpolation='nearest', cmap=plt.cm.gray)
plt.title("Output B")
plt.colorbar()
plt.show()

# Plot the final output
plt.figure()
plt.imshow(out[0][0].detach().numpy(), interpolation='nearest', cmap=plt.cm.gray)
plt.title("Final Output = outA + outB")
plt.colorbar()
plt.show()
