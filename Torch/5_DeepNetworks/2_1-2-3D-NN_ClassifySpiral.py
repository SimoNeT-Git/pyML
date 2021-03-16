#!/usr/bin/env python
# coding: utf-8

# In this lab, you will create a Deeper Neural Network with nn.ModuleList()
#   - Neural Network Module and Function for Training
#   - Train and Validate the Model
#   - Analyze Results

from time import time
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from matplotlib.colors import ListedColormap
from torch.utils.data import Dataset, DataLoader

torch.manual_seed(1)

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


# Use the following helper functions for plotting the loss:
def plot_accuracy_loss(training_results):

    fig, ax1 = plt.subplots()
    color = 'tab:red'
    ax1.plot(training_results['cost'], color=color)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Total Loss on Training Set', color=color)
    ax1.tick_params(axis='y', color=color)
    ax1.set_title('Cost / Accuracy after every Epoch')

    ax2 = ax1.twinx()
    color = 'tab:blue'
    ax2.plot(training_results['accuracy'], color=color)
    ax2.set_ylabel('Accuracy on whole DataSet', color=color)
    ax2.tick_params(axis='y', color=color)
    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.show()


# Define the function to plot the diagram
def plot_decision_regions_3class(data_set, model):
    cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#00AAFF'])
    # cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#00AAFF'])
    X = data_set.x.numpy()
    Y = data_set.y.numpy()
    h = .02
    x_min, x_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
    y_min, y_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    XX = torch.Tensor(np.c_[xx.ravel(), yy.ravel()])
    _, yhat = torch.max(model(XX.to(device)), 1)
    yhat = yhat.cpu().numpy().reshape(xx.shape)
    plt.figure()
    plt.pcolormesh(xx, yy, yhat, cmap=cmap_light)
    plt.plot(X[Y[:] == 0, 0], X[Y[:] == 0, 1], 'ro', label='y = 0')
    plt.plot(X[Y[:] == 1, 0], X[Y[:] == 1, 1], 'go', label='y = 1')
    plt.plot(X[Y[:] == 2, 0], X[Y[:] == 2, 1], 'bo', label='y = 2')
    plt.title("Decision Region")
    plt.legend()
    plt.show()


# Create Data Class: modified from: http://cs231n.github.io/neural-networks-case-study/
class Data(Dataset):

    # Constructor
    def __init__(self, K=3, N=500):
        D = 2
        X = np.zeros((N * K, D))  # data matrix (each row = single example)
        y = np.zeros(N * K, dtype='uint8')  # class labels
        for j in range(K):
            ix = range(N * j, N * (j + 1))
            r = np.linspace(0.0, 1, N)  # radius
            t = np.linspace(j * 4, (j + 1) * 4, N) + np.random.randn(N) * 0.2  # theta
            X[ix] = np.c_[r * np.sin(t), r * np.cos(t)]
            y[ix] = j
        self.y = torch.from_numpy(y).type(torch.LongTensor)
        self.x = torch.from_numpy(X).type(torch.FloatTensor)
        self.len = y.shape[0]

    # Getter
    def __getitem__(self, index):
        return self.x[index], self.y[index]

    # Get Length
    def __len__(self):
        return self.len

    # Plot the diagram
    def plot_stuff(self):
        plt.figure()
        plt.plot(self.x[self.y[:] == 0, 0].numpy(), self.x[self.y[:] == 0, 1].numpy(), 'ro', label="y = 0")
        plt.plot(self.x[self.y[:] == 1, 0].numpy(), self.x[self.y[:] == 1, 1].numpy(), 'go', label="y = 1")
        plt.plot(self.x[self.y[:] == 2, 0].numpy(), self.x[self.y[:] == 2, 1].numpy(), 'bo', label="y = 2")
        plt.legend()
        plt.title('Data Points')
        plt.show()


# #### Neural Network Module and Function for Training

# Create Net model class using ModuleList()
class Net(nn.Module):

    # Constructor
    def __init__(self, Layers):
        super(Net, self).__init__()
        self.hidden = nn.ModuleList()
        for input_size, output_size in zip(Layers, Layers[1:]):
            self.hidden.append(nn.Linear(input_size, output_size))

    # Prediction
    def forward(self, activation):
        L = len(self.hidden)
        for (l, linear_transform) in zip(range(L), self.hidden):
            if l < L - 1:
                activation = torch.relu(linear_transform(activation))
            else:
                activation = linear_transform(activation)
        return activation


# Define the function for training the model
def train(train_loader, data_set, model, criterion, optimizer, epochs=100):
    training_results = {'cost': [], 'accuracy': []}
    for epoch in range(epochs):
        tot_loss = 0
        for x, y in train_loader:
            optimizer.zero_grad()
            yhat = model(x.to(device))
            loss = criterion(yhat, y.to(device))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            tot_loss += loss.item()
        training_results['cost'].append(tot_loss / len(train_loader))
        training_results['accuracy'].append(accuracy(model, data_set))
    return training_results


# The function to calculate the accuracy
def accuracy(model, data_set):
    _, yhat = torch.max(model(data_set.x.to(device)), 1)
    return (yhat == data_set.y.to(device)).cpu().numpy().mean() * 100


# #### Train and Validate the Model

# Create a Dataset object
data_set = Data()
data_set.plot_stuff()
data_set.y = data_set.y.view(-1)
train_loader = DataLoader(dataset=data_set, batch_size=20)

# Create the criterion object
criterion = nn.CrossEntropyLoss()

# Set fix parameters
lr = 0.01
epochs = 1000


## 1-D Neural Network
# Create a network to classify three classes with 1 hidden layer with 30 neurons
Layers_1D = [2, 30, 3]
model_1D = Net(Layers_1D).to(device)
# Train the model with 1 hidden layer with 30 neurons
optimizer_1D = torch.optim.SGD(model_1D.parameters(), lr=lr)
print("\nLets train the Neural Network with 1 hidden layer having 30 neurons.")
t0 = time()
training_results_1D = train(train_loader, data_set, model_1D, criterion, optimizer_1D, epochs=epochs)
print("The whole training process, which consists of", epochs, "epochs, took", round(time()-t0, 2), "seconds.")
plot_decision_regions_3class(data_set, model_1D)
plot_accuracy_loss(training_results_1D)


## 2-D Neural Network
# Create a network to classify three classes with 2 hidden layers with 30 neurons in total
Layers_2D = [2, 15, 15, 3]
model_2D = Net(Layers_2D).to(device)
# Net([3, 3, 4, 3]).parameters
# Train the model with 2 hidden layers with 30 neurons
optimizer_2D = torch.optim.SGD(model_2D.parameters(), lr=lr)
print("\nLets train the Neural Network with 2 hidden layers having 15 neurons each.")
t0 = time()
training_results_2D = train(train_loader, data_set, model_2D, criterion, optimizer_2D, epochs=epochs)
print("The whole training process, which consists of", epochs, "epochs, took", round(time()-t0, 2), "seconds.")
plot_decision_regions_3class(data_set, model_2D)
plot_accuracy_loss(training_results_2D)


## 3-D Neural Network
# Create a network to classify three classes with 3 hidden layers with 30 neurons in total
Layers_3D = [2, 10, 10, 10, 3]
model_3D = Net(Layers_3D).to(device)
# Train the model with 3 hidden layers with 30 neurons
optimizer_3D = torch.optim.SGD(model_3D.parameters(), lr=lr)
print("\nLets train the Neural Network with 3 hidden layers having 10 neurons each.")
t0 = time()
training_results_3D = train(train_loader, data_set, model_3D, criterion, optimizer_3D, epochs=epochs)
print("The whole training process, which consists of", epochs, "epochs, took", round(time()-t0, 2), "seconds.")
plot_decision_regions_3class(data_set, model_3D)
plot_accuracy_loss(training_results_3D)


# #### Analyze Results

# Compare the training loss for each activation
plt.figure()
plt.plot(training_results_1D['cost'], label='1D-NN')
plt.plot(training_results_2D['cost'], label='2D-NN')
plt.plot(training_results_3D['cost'], label='3D-NN')
plt.xlabel('Epoch')
plt.ylabel('Cost')
plt.title('Total-Loss on Training Set at each Epoch')
plt.legend()
plt.show()

# Print max accuracy of all models
print("\nLets look at the maximum accuracy value achieved with the 3 models tested.")
print("  - 1D-NN:", round(max(training_results_1D['accuracy']), 2), "%")
print("  - 2D-NN:", round(max(training_results_2D['accuracy']), 2), "%")
print("  - 3D-NN:", round(max(training_results_3D['accuracy']), 2), "%")

# Compare the validation loss for each model
plt.figure()
plt.plot(training_results_1D['accuracy'], label='1D-NN')
plt.plot(training_results_2D['accuracy'], label='2D-NN')
plt.plot(training_results_3D['accuracy'], label='3D-NN')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Accuracy on whole DataSet at each Epoch')
plt.legend()
plt.show()
