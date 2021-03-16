#!/usr/bin/env python
# coding: utf-8

# In this lab, we will use a Convolutional Neural Network to classify handwritten digits from the MNIST database.
# We will reshape the images to make them faster to process.
#   - Get Some Data
#   - Convolutional Neural Network Class and Training Function
#   - Define Softmax, Criterion function, Optimizer and Train the Model
#   - Analyze Results

from time import time
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as dsets
import matplotlib.pylab as plt
from sklearn.metrics import confusion_matrix
import numpy as np
import itertools

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


# One way of looking at accuracy of classifier is to look at the confusion matrix.
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = np.round(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100, 0)

    plt.figure()
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(int(cm[i, j])) + (' %' if normalize else ''),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()


# Define the function show_data to plot out data samples as images.
def show_data(data_sample, prediction=None):
    size = data_sample[0].shape[1]
    plt.figure()
    plt.imshow(data_sample[0].cpu().numpy().reshape(size, size), cmap='gray')
    plt.suptitle('Data sample with label y = ' + str(data_sample[1]))
    if prediction is not None:
        plt.title('$\Rightarrow$ Predicted label is $\hat{y}$ = ' + str(prediction))
    plt.axis('off')
    plt.show()


# Define the function plot_channels to plot out the kernel parameters of each channel
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


# Define the function plot_parameters to plot out the kernel parameters of each channel with Multiple outputs.
def plot_parameters(W, number_rows=1, name="", i=0):
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    W = W.data[:, i, :, :].cpu()
    n_filters = W.shape[0]

    fig, axes = plt.subplots(number_rows, n_filters // number_rows,
                             figsize=(n_filters // number_rows * 6, number_rows * 4))
    fig.subplots_adjust(hspace=0.4)
    for i, ax in enumerate(axes.flat):
        if i < n_filters:
            # Plot the image.
            img = ax.imshow(W[i, :], vmin=-1., vmax=1., cmap='seismic')
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="3%", add_to_figure=False)
            plt.colorbar(img, ax=cax)
            ax.axis('off')
            # Set the label for the sub-plot.
            ax.set_xlabel("kernel {0}".format(i + 1))

    plt.suptitle(name, fontsize=10)
    plt.show()


# Define the function plot_activation to plot out the activations of the Convolutional layers
def plot_activations(A, number_rows=1, name=""):
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    A = A[0, :, :, :].detach().cpu().numpy()
    n_activations = A.shape[0]

    A_min = A.min().item()
    A_max = A.max().item()

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


# #### Get the Data

# we create a transform to resize the image and convert it to a tensor.
IMAGE_SIZE = 16
composed = transforms.Compose([transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)), transforms.ToTensor()])

# Load the training dataset by setting the parameters train to True. We use the transform defined above.
train_dataset = dsets.MNIST(root='./Data', train=True, download=True, transform=composed)

# Load the testing dataset by setting the parameters train False.
validation_dataset = dsets.MNIST(root='./Data', train=False, download=True, transform=composed)

# Create training and validation set determining the batch size for each
kwargs_loader = {'num_workers': 4 * (current_dev + 1), 'pin_memory': True} if use_cuda else {}
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=100, **kwargs_loader)
validation_loader = torch.utils.data.DataLoader(dataset=validation_dataset, batch_size=5000, **kwargs_loader)

# Each element in the rectangular tensor corresponds to a number representing a pixel intensity.
# The image for the first data element
show_data(train_dataset[0])
# This sample is a "5".


# #### Build a Convolutional Neural Network Class and the Training Function

# Build a Convolutional Network class with two Convolutional layers and one fully connected layer. Pre-determine the
# size of the final output matrix. The parameters in the constructor are the number of output channels for the first and
# second layer.
class CNN(nn.Module):

    # Constructor
    def __init__(self, out_1=1, out_2=2, out_dim=10, n_classes=10, ks_conv=2, ks_pool=2, stride_conv=1, stride_pool=1, padding=1):
        super(CNN, self).__init__()

        # first Convolutional layer
        self.cnn1 = nn.Conv2d(in_channels=1, out_channels=out_1, kernel_size=ks_conv, padding=padding, stride=stride_conv)
        # first Pooling layer
        self.maxpool1 = nn.MaxPool2d(kernel_size=ks_pool, stride=stride_pool)

        # second Convolutional layer
        self.cnn2 = nn.Conv2d(in_channels=out_1, out_channels=out_2, kernel_size=ks_conv, padding=padding, stride=stride_conv)
        # second Pooling layer
        self.maxpool2 = nn.MaxPool2d(kernel_size=ks_pool, stride=stride_pool)

        # final, fully connected, layer
        self.fc1 = nn.Linear(out_2 * out_dim * out_dim, n_classes)

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
        # outputs activation this is not necessary
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
            # transfer data to gpu if available
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
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
            # add each loss to compute total-loss/cost
            cost += loss.item()
        # compute the cost at every epoch
        cost_list.append(cost / len(train_loader))

        # perform a prediction on the validation data
        correct = 0
        for x_test, y_test in validation_loader:
            # transfer data to gpu if available
            x_test, y_test = x_test.to(device, non_blocking=True), y_test.to(device, non_blocking=True)
            # make prediction
            z = model(x_test)
            _, yhat = torch.max(z.data, 1)
            # add each correct prediction
            correct += (yhat == y_test).sum().item()
        # compute accuracy at every epoch
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
ks_conv = 5
ks_pool = 2
stride_conv = 1
stride_pool = 2
padding = 2

# The input image is 16 x 16, but after each convolutional/pooling layer the size of the activation (the output of that
# layer, thus the input of the subsequent one) will change. Knowing the parameters of each convolution/pooling layer
# (padding, kernel size, stride and dilation) we can determine the size of their outputs thanks to the conv_output_shape
# function.
in_size = (IMAGE_SIZE, IMAGE_SIZE)
out1 = conv_output_shape(in_size, kernel_size=ks_conv, stride=stride_conv, pad=padding)
out2 = conv_output_shape(out1, kernel_size=ks_pool, stride=stride_pool)
out3 = conv_output_shape(out2, kernel_size=ks_conv, stride=stride_conv, pad=padding)
out4 = conv_output_shape(out3, kernel_size=ks_pool, stride=stride_pool)

# There are 16 output channels for the first layer, and 32 output channels for the second layer.
# Create the model object using CNN class.
model = CNN(out_1=16, out_2=16*2, out_dim=out4[0], n_classes=len(train_dataset.classes),
            ks_conv=ks_conv, ks_pool=ks_pool,
            stride_conv=stride_conv, stride_pool=stride_pool, padding=padding).to(device)

# we can see the model parameters with the object
print('\nLets print out information about the Convolutional Neural Network we are building:')
print(model)
print('\nThe shape of the input images is:                                   ', in_size)
print('After the first convolutional layer, the shape of the output is:    ', out1)
print('After the first pooling layer, the shape of the output is:            ', out2)
print('After the second convolutional layer, the shape of the output is:     ', out3)
print('Finally, after the second pooling layer, the shape of the output is:  ', out4)

# Plot the model parameters for the kernels before training the kernels. The kernels are initialized randomly.
plot_parameters(model.state_dict()['cnn1.weight'], number_rows=out4[0],
                name='Initialization kernels of the first convolutional layer')
plot_parameters(model.state_dict()['cnn2.weight'], number_rows=out4[0],
                name='Initialization kernels of the second convolutional layer')

# Define the loss function, the optimizer and the dataset loader
criterion = nn.CrossEntropyLoss()
learning_rate = 0.1
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# Train the model and determine validation accuracy technically test accuracy **(This may take a long time)**
print("\nNow lets train the Convolutional Neural Network. This will take some time.")
n_epochs = 6
t0 = time()
cost_list, accuracy_list = train(train_loader, validation_loader, model, criterion, optimizer, n_epochs)
print("The whole training process, which consists of", n_epochs, "epochs, took", round(time()-t0, 2), "seconds.")


# #### Analyze Results

# Plot the kernels in both convolutional layers after the training process.
plot_parameters(model.state_dict()['cnn1.weight'], number_rows=out4[0],
                name='After-training kernels of the first convolutional layer')
plot_parameters(model.state_dict()['cnn2.weight'], number_rows=out4[0],
                name='After-training kernels of the second convolutional layer')

# Print some information
print('\nThe maximum accuracy value reached on the validation set is:', round(accuracy_list[-1], 2), '%')

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

# View the results of the parameters for the Convolutional layers
plot_channels(model.state_dict()['cnn1.weight'], name='After-training kernels of the first convolutional layer')
plot_channels(model.state_dict()['cnn2.weight'], name='After-training kernels of the second convolutional layer')

# Determine the activations (outputs of each layer) for input image 0 (label 5) and 1 (label 0)
out0_cnn1, out0_cnn1_relu, out0_cnn2, out0_cnn2_relu, out0_final = model.activations(
    train_dataset[0][0].view(1, 1, IMAGE_SIZE, IMAGE_SIZE).to(device))
out1_cnn1, out1_cnn1_relu, out1_cnn2, out1_cnn2_relu, out1_final = model.activations(
    train_dataset[1][0].view(1, 1, IMAGE_SIZE, IMAGE_SIZE).to(device))

# Plot the outputs after the first CNN
plot_activations(out0_cnn1_relu, number_rows=4,
                 name="Output of first convolutional layer + ReLu\nfor image with label {}".format(train_dataset[0][1]))
plot_activations(out0_cnn2_relu, number_rows=4,
                 name="Output of second convolutional layer + ReLu\nfor image with label {}".format(train_dataset[0][1]))
plot_activations(out0_final.view(1, 16*2, out4[0], out4[0]), number_rows=4,
                 name="Final output for image with label {}".format(train_dataset[0][1]))
plot_activations(out1_cnn1_relu, number_rows=4,
                 name="Output of first convolutional layer + ReLu\nfor image with label {}".format(train_dataset[1][1]))
plot_activations(out1_cnn2_relu, number_rows=4,
                 name="Output of second convolutional layer + ReLu\nfor image with label {}".format(train_dataset[1][1]))
plot_activations(out1_final.view(1, 16*2, out4[0], out4[0]), number_rows=4,
                 name="Final output for image with label {}".format(train_dataset[1][1]))

# Plot the first five mis-classified samples:
count = 0
test = torch.utils.data.DataLoader(dataset=validation_dataset, batch_size=1, **kwargs_loader)
for k, (x, y) in enumerate(test):
    x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
    z = model(x)
    _, yhat = torch.max(z, 1)
    if yhat != y:
        show_data(validation_dataset[k], prediction=yhat.item())
        count += 1
    if count >= 5:
        break

# Generate and plot the confusion matrix
test_set = torch.utils.data.DataLoader(dataset=validation_dataset, batch_size=1, **kwargs_loader)
y_hat = []
model.eval()
for x, y in test_set:
    x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
    z = model(x)
    _, yhat = torch.max(z.data, 1)
    y_hat.append(yhat.item())
y_hat = np.array(y_hat)
y_test = test_set.sampler.data_source.targets.numpy()
cnf_matrix = confusion_matrix(y_test, y_hat, labels=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
plot_confusion_matrix(cnf_matrix, classes=['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'],
                      normalize=True, title='Confusion matrix for CNN without Batch Normalization')
