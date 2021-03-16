#!/usr/bin/env python
# coding: utf-8

# In this lab, we will use a Convolutional Neural Networks to classify horizontal an vertical Lines.
#   - Helper functions
#   - Prepare and Visualize Data
#   - Build a Convolutional Neural Network Class and the Training Function
#   - Define Softmax, Criterion function, Optimizer and Train the Model
#   - Analyse Results

from time import time
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import matplotlib.pylab as plt
import numpy as np

torch.manual_seed(4)

# Use GPU if possible! Note: for small models like this one there will be no speed-up of the training process.
use_cuda = torch.cuda.is_available()
if use_cuda:  # Print additional info when using cuda
    device = torch.device('cuda')
    current_dev = torch.cuda.current_device()
    cap_dev = torch.cuda.get_device_capability()
    print('\nRunning on GPU', torch.cuda.get_device_name(device),
          '(compute capability {}.{})'.format(cap_dev[0], cap_dev[1]),
          'number', current_dev + 1, 'of', torch.cuda.device_count())
    print('Total memory:     ', round(torch.cuda.get_device_properties(current_dev).total_memory / 1024 ** 2, 1), 'MB')
    print('Allocated memory: ', round(torch.cuda.memory_allocated(current_dev) / 1024 ** 2, 1), 'MB')
    print('Cached memory:    ', round(torch.cuda.memory_cached(current_dev) / 1024 ** 2, 1), 'MB\n')
else:
    device = torch.device('cpu')
    print('\nRunning on CPU.\n')


# #### Helper functions

# create some toy data
class Data(Dataset):
    def __init__(self, N_images=100, offset=0, p=0.9, train=False):
        """
        p:portability that pixel is wight  
        N_images:number of images 
        offset:set a random vertical and horizontal offset images by a sample should be less than 3 
        """
        if train:
            np.random.seed(1)

        N_images = 2 * (N_images // 2)  # make images multiple of 3
        images = np.zeros((N_images, 1, 11, 11))
        start1 = 3
        start2 = 1
        self.y = torch.zeros(N_images).type(torch.long).to(device)

        for n in range(N_images):
            if offset > 0:
                low = int(np.random.randint(low=start1, high=start1 + offset, size=1))
                high = int(np.random.randint(low=start2, high=start2 + offset, size=1))
            else:
                low = 4
                high = 1

            if n <= N_images // 2:
                self.y[n] = 0
                images[n, 0, high:high + 9, low:low + 3] = np.random.binomial(1, p, (9, 3))
            elif n > N_images // 2:
                self.y[n] = 1
                images[n, 0, low:low + 3, high:high + 9] = np.random.binomial(1, p, (3, 9))

        self.x = torch.from_numpy(images).type(torch.FloatTensor).to(device)
        self.len = self.x.shape[0]
        del (images)
        np.random.seed(0)

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return self.len


# show_data: plot out data sample
def show_data(dataset, sample):
    plt.figure()
    plt.imshow(dataset.x[sample, 0, :, :].cpu().numpy(), cmap='gray')
    plt.title('y=' + str(dataset.y[sample].item()))
    plt.show()


# function to plot out the parameters of the Convolutional layers
def plot_channels(W, name=''):
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    W = W.cpu()
    n_out = W.shape[0]  # number of output channels
    n_in = W.shape[1]  # number of input channels

    fig, axes = plt.subplots(n_out, n_in, figsize=(n_in * 6, n_out * 4))
    fig.subplots_adjust(hspace=0.1)
    out_index = 0
    in_index = 0
    # plot outputs as rows, inputs as columns
    for ax in axes.flat:

        if in_index > n_in - 1:
            out_index = out_index + 1
            in_index = 0

        img = ax.imshow(W[out_index, in_index, :, :], vmin=-1., vmax=1., cmap='seismic')
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="3%", add_to_figure=False)
        plt.colorbar(img, ax=cax)
        ax.axis('off')
        in_index = in_index + 1

    fig.suptitle(name + '\n' + str(n_in) + ' input and ' + str(n_out) + ' output channels', y=1.0009)
    plt.show()


# plot_activation: plot out the activations of the Convolutional layers
def plot_activations(A, number_rows=1, name=''):
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    A = A[0, :, :, :].detach().cpu().numpy()
    n_activations = A.shape[0]

    A_min = A.min().item()
    A_max = A.max().item()

    if n_activations == 1:
        fig = plt.figure()
        img = plt.imshow(A[0, :], vmin=A_min, vmax=A_max, cmap='seismic')  # Plot the image.
        plt.colorbar(img)
    else:
        fig, axes = plt.subplots(number_rows, n_activations // number_rows,
                                 figsize=(n_activations // number_rows * 6, number_rows * 4))
        fig.subplots_adjust(hspace=0.4)
        for i, ax in enumerate(axes.flat):
            if i < n_activations:
                ax.set_xlabel("channel {0}".format(i + 1))  # Set the label for the sub-plots
                img = ax.imshow(A[i, :], vmin=A_min, vmax=A_max, cmap='seismic')  # Plot the image.
                divider = make_axes_locatable(ax)
                cax = divider.append_axes("right", size="3%", add_to_figure=False)
                plt.colorbar(img, ax=cax)

    fig.suptitle(name)
    plt.show()


# Utility function for computing output of convolutions takes a tuple of (h,w) and returns a tuple of (h,w)
def conv_output_shape(h_w, kernel_size=1, stride=1, pad=0, dilation=1):
    # by Duane Nielsen
    from math import floor
    if type(kernel_size) is not tuple:
        kernel_size = (kernel_size, kernel_size)
    h = floor(((h_w[0] + (2 * pad) - (dilation * (kernel_size[0] - 1)) - 1) / stride) + 1)
    w = floor(((h_w[1] + (2 * pad) - (dilation * (kernel_size[1] - 1)) - 1) / stride) + 1)
    return h, w


# #### Prepare and Visualize Data

# Load the training dataset with 10000 samples. Note that from sample 0 to N_images/2 = 5000 the label will be y=0
# (vertical line), while from N_images/2 + 1 = 5001 to N_images - 1 = 9999 the label will be y=1 (horizontal line).
N_images = 10000
train_dataset = Data(N_images=N_images)
# Each element in the rectangular tensor corresponds to a number representing a pixel intensity.
# Lets print out some labels
fig = plt.figure(figsize=(8, 4))
plt.subplot(121)
plt.imshow(train_dataset.x[0, 0, :, :].cpu().numpy(), cmap='gray')
plt.axis('off')
plt.title('y=' + str(train_dataset.y[0].item()))
plt.subplot(122)
plt.imshow(train_dataset.x[N_images // 2 + 1, 0, :, :].cpu().numpy(), cmap='gray')
plt.axis('off')
plt.title('y=' + str(train_dataset.y[N_images // 2 + 1].item()))
fig.suptitle('2 Samples in the Training Set')
plt.show()

# Load the testing dataset with 1000 samples. Note that from sample 0 to 500 the label will be y=0 (vertical line),
# while from 501 to 999 the label will be y=1 (horizontal line).
validation_dataset = Data(N_images=1000, train=False)

# Create training and validation set determining the batch size for each
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=10)
validation_loader = torch.utils.data.DataLoader(dataset=validation_dataset, batch_size=20)

# ### Build a Convolutional Neural Network Class and the Training Function


# Build a Convolutional Network class with two Convolutional layers and one fully connected layer. Pre-determine the
# size of the final output matrix. The parameters in the constructor are the number of output channels for the first and
# second layer.
class CNN(nn.Module):

    # Constructor
    def __init__(self, out_1=2, out_2=1, out_dim=7, num_classes=2, kernel_size=2, stride=1, padding=0):
        super(CNN, self).__init__()

        # first Convolutional layer
        self.cnn1 = nn.Conv2d(in_channels=1, out_channels=out_1, kernel_size=kernel_size, padding=padding)
        # first Pooling layer
        self.maxpool1 = nn.MaxPool2d(kernel_size=kernel_size, stride=stride)

        # second Convolutional layer
        self.cnn2 = nn.Conv2d(in_channels=out_1, out_channels=out_2, kernel_size=kernel_size, stride=stride,
                              padding=padding)
        # second Pooling layer
        self.maxpool2 = nn.MaxPool2d(kernel_size=kernel_size, stride=stride)

        # final, fully connected, layer
        self.fc1 = nn.Linear(out_2 * out_dim * out_dim, num_classes)

    # Prediction
    def forward(self, x):
        # first Convolutional layers
        x = self.cnn1(x)
        # activation function
        x = torch.relu(x)
        # max pooling
        x = self.maxpool1(x)
        # first Convolutional layers
        x = self.cnn2(x)
        # activation function
        x = torch.relu(x)
        # max pooling
        x = self.maxpool2(x)
        # flatten output
        x = x.view(x.size(0), -1)
        # fully connected layer
        x = self.fc1(x)
        return x

    # Outputs in each layer
    def activations(self, x):
        # outputs activation; this is not necessary, just for fun.
        z1 = self.cnn1(x)
        a1 = torch.relu(z1)
        out = self.maxpool1(a1)

        z2 = self.cnn2(out)
        a2 = torch.relu(z2)
        out = self.maxpool2(a2)
        out = out.view(out.size(0), -1)
        return z1, a1, z2, a2, out


# Define tha function for training the network.
def train(train_loader, validation_loader, model, criterion, optimizer, n_epochs):
    cost_list = []
    accuracy_list = []

    for epoch in range(n_epochs):
        cost = 0
        for x, y in train_loader:
            # clear gradient
            optimizer.zero_grad()
            # make a prediction
            z = model(x)
            # calculate loss
            loss = criterion(z, y)
            # calculate gradients of parameters
            loss.backward()
            # update parameters
            optimizer.step()
            cost += loss.item()
        cost_list.append(cost / len(train_loader))

        correct = 0
        # perform a prediction on the validation  data
        for x_test, y_test in validation_loader:
            z = model(x_test)
            _, yhat = torch.max(z.data, 1)
            correct += (yhat == y_test).sum().item()

        accuracy = correct / (len(validation_loader) * validation_loader.batch_size) * 100
        accuracy_list.append(accuracy)
    return cost_list, accuracy_list


# #### Define the Convolutional Neural Network Classifier, Criterion function, Optimizer and Train the Model

# The structure of the nework is the following:
# - convolutional layer 1
# - max pooling layer 1
# - convolutional layer 2
# - max pooling layer 2
# all with the following parameters' values.
kernel_size = 2
stride = 1
padding = 0

# The input image is 11 x 11, but after each convolutional/pooling layer the size of the activation (the output of that
# layer, thus the input of the subsequent one) will change. Knowing the parameters of each convolution/pooling layer
# (padding, kernel size, stride and dilation) we can determine the size of their outputs thanks to the conv_output_shape
# function.
in_size = (11, 11)
out1 = conv_output_shape(in_size, kernel_size=kernel_size, stride=stride, pad=padding, dilation=1)
out2 = conv_output_shape(out1, kernel_size=kernel_size, stride=stride, pad=padding, dilation=1)
out3 = conv_output_shape(out2, kernel_size=kernel_size, stride=stride, pad=padding, dilation=1)
out4 = conv_output_shape(out3, kernel_size=kernel_size, stride=stride, pad=padding, dilation=1)

# There are 2 output channels for the first layer, and 1 outputs channel for the second layer
model = CNN(out_1=2, out_2=1, out_dim=out4[0], num_classes=2,
            kernel_size=kernel_size, stride=stride, padding=padding).to(device)

# we can see the model parameters with the object
print('\nLets print out information about the Convolutional Neural Network we are building:')
print(model)
print('\nThe shape of the input images is:                                   ', in_size)
print('After the first convolutional layer, the shape of the output is:    ', out1)
print('After the first pooling layer, the shape of the output is:            ', out2)
print('After the second convolutional layer, the shape of the output is:     ', out3)
print('Finally, after the second pooling layer, the shape of the output is:  ', out4)

# Plot the model parameters for the kernels before training the kernels. The kernels are initialized randomly.
plot_channels(model.state_dict()['cnn1.weight'], name='Initialization kernels of the first convolutional layer')
plot_channels(model.state_dict()['cnn2.weight'], name='Initialization kernels of the second convolutional layer')

# Define the loss function
criterion = nn.CrossEntropyLoss()

# Define the optimizer class
learning_rate = 0.001
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Train the model and determine validation accuracy (technically test accuracy) **(This may take a long time)**
print("\nNow lets train the Convolutional Neural Network. This will take some time.")
n_epochs = 10
t0 = time()
cost_list, accuracy_list = train(train_loader, validation_loader, model, criterion, optimizer, n_epochs)
print("The whole training process, which consists of", n_epochs, "epochs, took", round(time()-t0, 2), "seconds.")


# #### Analyze Results

# View the results of the parameters for the Convolutional layers
plot_channels(model.state_dict()['cnn1.weight'], name='After-training kernels of the first convolutional layer')
plot_channels(model.state_dict()['cnn2.weight'], name='After-training kernels of the second convolutional layer')

# Print some information
print('\nThe maximum accuracy value reached on the validation set is:', accuracy_list[-1], '%')

# Plot the loss and accuracy on the validation data:
fig, ax1 = plt.subplots(figsize=(8, 6))
color = 'tab:red'
ax1.plot(cost_list, color=color)
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Total Loss', color=color)
ax1.tick_params(axis='y', color=color)

ax2 = ax1.twinx()
color = 'tab:blue'
ax2.set_ylabel('Accuracy', color=color)
ax2.plot(accuracy_list, color=color)
ax2.tick_params(axis='y', labelcolor=color)
fig.tight_layout()
fig.suptitle('Cost on Training Set and Accuracy on Validation Set')
plt.show()

# Determine the activations (outputs of each layer) for input images with label 0 and 1
out0_cnn1, out0_cnn1_relu, out0_cnn2, out0_cnn2_relu, out0_final = model.activations(
    train_dataset[0][0].view(1, 1, 11, 11))
out1_cnn1, out1_cnn1_relu, out1_cnn2, out1_cnn2_relu, out1_final = model.activations(
    train_dataset[N_images // 2 + 2][0].view(1, 1, 11, 11))

# Plot them out
plot_activations(out0_cnn1_relu, number_rows=1,
                 name="Output of first convolutional layer + ReLu\nfor image with label 0")
plot_activations(out0_cnn2_relu, number_rows=1,
                 name="Output of second convolutional layer + ReLu\nfor image with label 0")
plot_activations(out0_final.view(1, 1, out4[0], out4[0]), number_rows=1,
                 name="Final Output for image with label 0")
plot_activations(out1_cnn1_relu, number_rows=1,
                 name="Output of first convolutional layer + ReLu\nfor image with label 1")
plot_activations(out1_cnn2_relu, number_rows=1,
                 name="Output of second convolutional layer + ReLu\nfor image with label 1")
plot_activations(out1_final.view(1, 1, out4[0], out4[0]), number_rows=1,
                 name="Final Output for image with label 1")

# Plot final flatted outputs
plt.figure()
plt.plot(out0_final[0].cpu().detach().numpy(), 'b', label='y=0')
plt.plot(out1_final[0].cpu().detach().numpy(), 'r', label='y=1')
plt.legend()
plt.xlabel('Index')
plt.ylabel('Activation')
plt.title('Flatted Activation Values')
plt.show()
