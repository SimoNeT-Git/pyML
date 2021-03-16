#!/usr/bin/env python
# coding: utf-8

# This lab takes a long time to run so the results are given. You can run the notebook yourself but it may take a
# long time.
# In this lab, we will compare a Convolutional Neural Network using Batch Normalization with a regular Convolutional
# Neural Network to classify handwritten digits from the MNIST database. We will reshape the images to make them faster
# to process.
#   - Read me Batch Norm for Convolution Operation
#   - Get Some Data
#   - Two Types of Convolutional Neural Networks (with and without Batch Normalization) and Training Function
#   - Define Criterion function, Optimizer and Train the Model
#   - Analyze Results

# #### Read me Batch Norm for Convolution Operation

# Like a fully connected network, we create a BatchNorm2d object, but we apply it to the 2D convolution object. First,
# we create objects Conv2d object; we require the number of output channels, specified by the variable OUT.

# self.cnn1 = nn.Conv2d(in_channels=1, out_channels=OUT, kernel_size=5, padding=2)

# We then create a Batch Norm  object for 2D convolution as follows:

# self.conv1_bn = nn.BatchNorm2d(OUT)

# The parameter out is the number of channels in the output. We can then apply batch norm  after the convolution
# operation:

# x = self.cnn1(x)
# x = self.conv1_bn(x)

# Import the libraries we need to use in this lab

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

# We create a transform to resize the image and convert it to a tensor:
IMAGE_SIZE = 16
composed = transforms.Compose([transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)), transforms.ToTensor()])

# Load the training dataset by setting the parameters train to True. We use the transform defined above.
train_dataset = dsets.MNIST(root='./Data', train=True, download=True, transform=composed)

# Load the testing dataset by setting the parameters train to False.
validation_dataset = dsets.MNIST(root='./Data', train=False, download=True, transform=composed)

# Each element in the rectangular tensor corresponds to a number representing a pixel intensity.
# The image for the first data element
show_data(train_dataset[0])
# This sample is a "5".

# Create training and validation set determining the batch size for each
kwargs_loader = {'num_workers': 4 * (current_dev + 1), 'pin_memory': True} if use_cuda else {}
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=100, **kwargs_loader)
validation_loader = torch.utils.data.DataLoader(dataset=validation_dataset, batch_size=5000, **kwargs_loader)


# #### Build Convolutional Neural Network (with and without Batch Normalization) Classes and the Training Function

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


# Build a Convolutional Network class with two Convolutional layers and one fully connected layer. But we add Batch Norm
# for the convolutional layers
class CNN_batch(nn.Module):

    # Constructor
    def __init__(self, out_1=1, out_2=2, out_dim=10, n_classes=10, ks_conv=2, ks_pool=2, stride_conv=1, stride_pool=1, padding=1):
        super(CNN_batch, self).__init__()

        # first Convolutional layer
        self.cnn1 = nn.Conv2d(in_channels=1, out_channels=out_1, kernel_size=ks_conv, padding=padding, stride=stride_conv)
        self.conv1_bn = nn.BatchNorm2d(out_1)
        # first Pooling layer
        self.maxpool1 = nn.MaxPool2d(kernel_size=ks_pool, stride=stride_pool)

        # second Convolutional layer
        self.cnn2 = nn.Conv2d(in_channels=out_1, out_channels=out_2, kernel_size=ks_conv, padding=padding, stride=stride_conv)
        self.conv2_bn = nn.BatchNorm2d(out_2)
        # second Pooling layer
        self.maxpool2 = nn.MaxPool2d(kernel_size=ks_pool, stride=stride_pool)

        # final, fully connected, layer
        self.fc1 = nn.Linear(out_2 * out_dim * out_dim, n_classes)
        self.bn_fc1 = nn.BatchNorm1d(n_classes)

    # Prediction
    def forward(self, x):
        x = self.cnn1(x)
        x = self.conv1_bn(x)
        x = torch.relu(x)
        x = self.maxpool1(x)
        x = self.cnn2(x)
        x = self.conv2_bn(x)
        x = torch.relu(x)
        x = self.maxpool2(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.bn_fc1(x)
        return x


# Define tha function for training the network.
def train(train_loader, validation_loader, model, criterion, optimizer, n_epochs):
    loss_list = []
    accuracy_list = []

    for epoch in range(n_epochs):
        cost = 0
        for x, y in train_loader:
            # transfer data to gpu if available
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            # set the model to training mode
            model.train()
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
            # compute the loss at every iteration
            loss_list.append(loss.item())

        # perform a prediction on the validation data
        correct = 0
        for x_test, y_test in validation_loader:
            # transfer data to gpu if available
            x_test, y_test = x_test.to(device, non_blocking=True), y_test.to(device, non_blocking=True)
            # set the model to evaluation mode
            model.eval()
            # make prediction
            z = model(x_test)
            _, yhat = torch.max(z.data, 1)
            # add each correct prediction
            correct += (yhat == y_test).sum().item()
        # compute accuracy at every epoch
        accuracy = correct / (len(validation_loader) * validation_loader.batch_size) * 100
        accuracy_list.append(accuracy)

    return loss_list, accuracy_list


# #### Define the Convolutional Neural Network Classifier, Criterion function, Optimizer and Train the Model

# The structure of both neworks is the following:
# - convolutional layer 1 (with or without batch normalization)
# - max pooling layer 1
# - convolutional layer 2 (with or without batch normalization)
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
# Create the 2 model objects using CNN and CNN_batch classes.
model = CNN(out_1=16, out_2=16*2, out_dim=out4[0], n_classes=len(train_dataset.classes),
            ks_conv=ks_conv, ks_pool=ks_pool,
            stride_conv=stride_conv, stride_pool=stride_pool, padding=padding).to(device)
model_batch = CNN_batch(out_1=16, out_2=16*2, out_dim=out4[0], n_classes=len(train_dataset.classes),
            ks_conv=ks_conv, ks_pool=ks_pool,
            stride_conv=stride_conv, stride_pool=stride_pool, padding=padding).to(device)

# we can see the models' parameters
print('\nLets print out information about the Convolutional Neural Network without batch normalization:')
print(model)
print('\nAnd those about the Convolutional Neural Network with batch normalization:')
print(model_batch)
print('\nIn both case we have:')
print('The shape of the input images is:                                   ', in_size)
print('After the first convolutional layer, the shape of the output is:    ', out1)
print('After the first pooling layer, the shape of the output is:            ', out2)
print('After the second convolutional layer, the shape of the output is:     ', out3)
print('Finally, after the second pooling layer, the shape of the output is:  ', out4)

# Define the loss function, the optimizer and the dataset loader 
criterion = nn.CrossEntropyLoss()
learning_rate = 0.1
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
optimizer_batch = torch.optim.SGD(model_batch.parameters(), lr=learning_rate)
n_epochs = 5

# Train the model and determine validation accuracy technically test accuracy **(This may take a long time)**
print("\nNow lets train the Convolutional Neural Network without batch normalization. This will take some time...")
t0 = time()
loss_list, accuracy_list = train(train_loader, validation_loader,
                                 model, criterion, optimizer, n_epochs)
print("The whole training process, which consists of", n_epochs, "epochs, took", round(time()-t0, 2), "seconds.")

# Repeat the Process for the model with batch norm
print("\nNow lets train the Convolutional Neural Network with batch normalization. This will take some time...")
t0 = time()
loss_list_batch, accuracy_list_batch = train(train_loader, validation_loader,
                                             model_batch, criterion, optimizer_batch, n_epochs)
print("The whole training process, which consists of", n_epochs, "epochs, took", round(time()-t0, 2), "seconds.")


# #### Analyze Results

# Print some information
print('\nThe maximum accuracy value reached on the validation set without batch normalization is:',
      round(accuracy_list[-1], 2), '%')
print('While that achieved with batch normalization is:', round(accuracy_list_batch[-1], 2), '%')

# Plot the loss with both networks.
plt.figure(figsize=(10, 4))
plt.subplot(121)
plt.plot(loss_list, 'b', label='normal CNN')
plt.plot(loss_list_batch, 'r', label='CNN with Batch Norm')
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.title("Loss on Training Set")
plt.legend()
plt.subplot(122)
plt.plot(accuracy_list, 'b', label='normal CNN')
plt.plot(accuracy_list_batch, 'r', label='CNN with Batch Norm')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title("Accuracy on Validation Set")
plt.legend()
plt.suptitle("Comparison of models' performances on MNIST dataset")
plt.show()
# We see the CNN with batch norm performers better, with faster convergence.

# Generate and plot the confusion matrix
test_set = torch.utils.data.DataLoader(dataset=validation_dataset, batch_size=1, **kwargs_loader)
model_batch.eval()
y_hat = []
y_hat_batch = []
model.eval()
model_batch.eval()
for x, y in test_set:
    x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
    z = model(x)
    z_batch = model_batch(x)
    _, yhat = torch.max(z.data, 1)
    _, yhat_batch = torch.max(z_batch.data, 1)
    y_hat.append(yhat.item())
    y_hat_batch.append(yhat_batch.item())
y_hat = np.array(y_hat)
y_hat_batch = np.array(y_hat_batch)
y_test = test_set.sampler.data_source.targets.numpy()
cnf_matrix = confusion_matrix(y_test, y_hat, labels=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
cnf_matrix_batch = confusion_matrix(y_test, y_hat_batch, labels=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
plot_confusion_matrix(cnf_matrix, classes=['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'],
                      normalize=True, title='Confusion matrix for CNN without Batch Normalization')
plot_confusion_matrix(cnf_matrix_batch, classes=['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'],
                      normalize=True, title='Confusion matrix for CNN with Batch Normalization')
