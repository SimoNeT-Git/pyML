#!/usr/bin/env python
# coding: utf-8

# In this lab, you will use a single layer Softmax to classify handwritten digits from the MNIST database.
#   - Make up Data
#   - Criterion function, Optimizer, and Train the Model
#   - Analyze Results

from time import time
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as dsets
import matplotlib.pylab as plt
import numpy as np


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


# Use the following function to plot out the parameters of the Softmax function:
def PlotParameters(model, trained=False):
    W = model.state_dict()['0.weight'].data.cpu()
    w_min = W.min().item()
    w_max = W.max().item()
    fig, axes = plt.subplots(2, 5, figsize=(15, 8))
    fig.subplots_adjust(hspace=0.01, wspace=0.1)
    for i, ax in enumerate(axes.flat):
        # Ensure the plot is shown correctly with multiple plots
        # in a single Notebook cell.
        if i < 10:
            # Set the label for the sub-plot.
            ax.set_title("class/label: {0}".format(i))
            ax.set_xlabel("w1")
            ax.set_ylabel("w2")

            # Plot the image.
            ax.imshow(W[i, :].view(28, 28), vmin=w_min, vmax=w_max, cmap='seismic')
            ax.set_xticks([])
            ax.set_yticks([])

    if trained:
        fig.suptitle("Weights' values for all classes after training")
    else:
        fig.suptitle("Weights' values for all classes at initialization")
    plt.show()


# Use the following function to visualize data:
def show_data(data_sample):
    plt.figure()
    plt.imshow(data_sample[0].numpy().reshape(28, 28), cmap='gray')
    plt.title('y = ' + str(data_sample[1]))  # label
    plt.show()


# #### Make Up Data

# Load the training dataset by setting the parameters train to True and convert it to a tensor by placing a transform
# object in the argument transform.
train_dataset = dsets.MNIST(root='./Data', train=True, download=True, transform=transforms.ToTensor())
print("\nTraining dataset information:\n ", train_dataset)

# Load the testing dataset by setting the parameters train to False and convert it to a tensor by placing a transform
# object in the argument transform.
validation_dataset = dsets.MNIST(root='./Data', train=False, download=True, transform=transforms.ToTensor())
print("\nValidation dataset information:\n ", validation_dataset)

# Note that the data type is long:
print("\nType of data element is: ", train_dataset[0][0].type())

# Print out the fourth label:
print("The labels of all", len(train_dataset), "samples, ranging from 0 to 9, are:", train_dataset.targets.tolist())

# Plot the the fourth sample:
print("\nAs an example, lets plot the fourth sample:")
show_data(train_dataset[3])
# You see its a 1.

# Now, plot the third sample:
print("\nOr the third one:")
show_data(train_dataset[2])

# The Softmax function requires vector inputs. If you see the vector shape, you'll note it's 28x28.
# Print the shape of the first element in train_dataset
print("\nThe shape of each sample in the dataset is: ", train_dataset[0][0].shape)

# Flatten the tensor: set the input size and output size.
input_dim = train_dataset[0][0].shape[1] * train_dataset[0][0].shape[2]  # = 28 * 28
output_dim = len(train_dataset.classes)  # = 10

# Create a Softmax Classifier by using Sequential
model = nn.Sequential(nn.Linear(input_dim, output_dim)).to(device)
print("\nModel info:\n ", model)

# View the size of the model parameters:
params = list(model.parameters())
print("\n\nThe randomly initialized weights of the model have shape", params[0].shape[0], "x", params[0].shape[1],
      "\nwhile the bias has shape", params[1].shape[0], "x 1")

# You can cover the model parameters for each class to a rectangular grid: plot the model parameters for each class as
# a square image:
PlotParameters(model, trained=False)


# #### Define the Softmax Classifier, Criterion function, Optimizer, and Train the Model

# Define the learning rate, optimizer, criterion and data loader
learning_rate = 0.1
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
criterion = nn.CrossEntropyLoss()
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=100)
validation_loader = torch.utils.data.DataLoader(dataset=validation_dataset, batch_size=5000)

# Train the model and determine validation accuracy:
t0 = time()
n_epochs = 10
loss_list = []
accuracy_list = []
N_test = len(validation_dataset)


def train_model(n_epochs):
    for epoch in range(n_epochs):
        # perform training (with parameter optimization) on training data
        for x, y in train_loader:
            optimizer.zero_grad()
            z = model(x.to(device).view(-1, 28 * 28))
            loss = criterion(z, y.to(device))
            loss.backward()
            optimizer.step()
        loss_list.append(loss.data)

        # perform a prediction on the validation data
        correct = 0
        for x_test, y_test in validation_loader:
            z = model(x_test.to(device).view(-1, 28 * 28))
            _, yhat = torch.max(z.data, 1)
            correct += (yhat == y_test.to(device)).sum().item()
        accuracy_list.append(correct / N_test)


train_model(n_epochs)
print("\nThe training process took", round((time() - t0), 2), "seconds")

# View the results of the parameters for each class after the training. You can see that they look like the
# corresponding numbers.
PlotParameters(model, trained=True)

# #### Analyze Results

# Plot the loss and accuracy on the validation data:
fig, ax1 = plt.subplots()
color = 'tab:red'
ax1.plot(loss_list, color=color)
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Total Loss', color=color)
ax1.tick_params(axis='y', color=color)

ax2 = ax1.twinx()
color = 'tab:blue'
ax2.set_ylabel('Accuracy', color=color)
ax2.plot(accuracy_list, color=color)
ax2.tick_params(axis='y', color=color)
fig.tight_layout()
plt.show()
