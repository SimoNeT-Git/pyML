#!/usr/bin/env python
# coding: utf-8

# In this lab, you will see how different values for the momentum parameters affect the convergence rate of a neural
# network.
#   - Neural Network Module and Function for Training
#   - Train Different Neural Networks Model different values for the Momentum Parameter
#   - Compare Results of Different Momentum Terms

from time import time
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from torch.utils.data import Dataset, DataLoader

torch.manual_seed(1)
np.random.seed(1)


# Use the following helper functions for plotting the loss:
def plot_accuracy_loss(training_results):

    fig, ax1 = plt.subplots()
    color = 'tab:red'
    ax1.plot(training_results['Cost'], color=color)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Total Loss on Training Set', color=color)
    ax1.tick_params(axis='y', color=color)
    ax1.set_title('Cost / Accuracy after every Epoch')

    ax2 = ax1.twinx()
    color = 'tab:blue'
    ax2.plot(training_results['Accuracy'], color=color)
    ax2.set_ylabel('Accuracy on whole DataSet', color=color)
    ax2.tick_params(axis='y', color=color)
    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.show()


# Define a function for plot the decision region
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
    _, yhat = torch.max(model(XX), 1)
    yhat = yhat.cpu().numpy().reshape(xx.shape)
    plt.figure()
    plt.pcolormesh(xx, yy, yhat, cmap=cmap_light)
    plt.plot(X[Y[:] == 0, 0], X[Y[:] == 0, 1], 'ro', label='y = 0')
    plt.plot(X[Y[:] == 1, 0], X[Y[:] == 1, 1], 'go', label='y = 1')
    plt.plot(X[Y[:] == 2, 0], X[Y[:] == 2, 1], 'bo', label='y = 2')
    plt.title("Decision Region")
    plt.legend()
    plt.show()


# Create the dataset class: modified from http://cs231n.github.io/neural-networks-case-study/
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
    def plot_data(self):
        plt.figure()
        plt.plot(self.x[self.y[:] == 0, 0].numpy(), self.x[self.y[:] == 0, 1].numpy(), 'ro', label="y = 0")
        plt.plot(self.x[self.y[:] == 1, 0].numpy(), self.x[self.y[:] == 1, 1].numpy(), 'go', label="y = 1")
        plt.plot(self.x[self.y[:] == 2, 0].numpy(), self.x[self.y[:] == 2, 1].numpy(), 'bo', label="y = 2")
        plt.legend()
        plt.title('Data Points')
        plt.show()


# #### Neural Network Module and Function for Training

# Create Neural Network Module using ModuleList()
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


# Create the function for training the model.
def train(train_loader, data_set, model, criterion, optimizer, epochs=100):
    training_results = {'Cost': [], 'Accuracy': []}
    for epoch in range(epochs):
        tot_loss = 0
        for x, y in train_loader:
            optimizer.zero_grad()
            yhat = model(x)
            loss = criterion(yhat, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            tot_loss += loss.item()
        training_results['Cost'].append(tot_loss / len(train_loader))
        training_results['Accuracy'].append(accuracy(model, data_set))
    return training_results


# Define a function used to calculate accuracy.
def accuracy(model, data_set):
    _, yhat = torch.max(model(data_set.x), 1)
    return (yhat == data_set.y).numpy().mean() * 100


# #### Train Different Networks Model different values for the Momentum Parameter

# Crate a dataset object using Data
data_set = Data()
data_set.plot_data()
data_set.y = data_set.y.view(-1)
train_loader = DataLoader(dataset=data_set, batch_size=20)

# Create the criterion object
criterion = nn.CrossEntropyLoss()

# Set fix parameters
Layers = [2, 50, 3]
lr = 0.1
epochs = 100

# Initialize a dictionary to contain different cost and accuracy values for each epoch for different values of the
# momentum parameter.
Results = {}

# Create a network to classify three classes with 1 hidden layer with 50 neurons and a momentum value of zero.
model = Net(Layers)
optimizer = torch.optim.SGD(model.parameters(), lr=lr)
print("\nLets train the Neural Network through optimizer with NO momentum.")
t0 = time()
Results["momentum 0"] = train(train_loader, data_set, model, criterion, optimizer, epochs=epochs)
print("The whole training process, which consists of", epochs, "epochs, took", round(time()-t0, 2), "seconds.")
print("Maximum accuracy value achieved is: ", round(Results["momentum 0"]['Accuracy'][-1], 1), "%")
plot_decision_regions_3class(data_set, model)
plot_accuracy_loss(Results["momentum 0"])

# Create a network to classify three classes with 1 hidden layer with 50 neurons and a momentum value of 0.1.
model = Net(Layers)
optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.1)
print("\nLets train the Neural Network through optimizer with momentum=0.1.")
t0 = time()
Results["momentum 0.1"] = train(train_loader, data_set, model, criterion, optimizer, epochs=epochs)
print("The whole training process, which consists of", epochs, "epochs, took", round(time()-t0, 2), "seconds.")
print("Maximum accuracy value achieved is: ", round(Results["momentum 0.1"]['Accuracy'][-1], 1), "%")
plot_decision_regions_3class(data_set, model)
plot_accuracy_loss(Results["momentum 0.1"])

# Create a network to classify three classes with 1 hidden layer with 50 neurons and a momentum value of 0.2.
model = Net(Layers)
optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.2)
print("\nLets train the Neural Network through optimizer with momentum=0.2.")
t0 = time()
Results["momentum 0.2"] = train(train_loader, data_set, model, criterion, optimizer, epochs=epochs)
print("The whole training process, which consists of", epochs, "epochs, took", round(time()-t0, 2), "seconds.")
print("Maximum accuracy value achieved is: ", round(Results["momentum 0.2"]['Accuracy'][-1], 1), "%")
plot_decision_regions_3class(data_set, model)
plot_accuracy_loss(Results["momentum 0.2"])

# Create a network to classify three classes with 1 hidden layer with 50 neurons and a momentum value of 0.3.
model = Net(Layers)
optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.3)
print("\nLets train the Neural Network through optimizer with momentum=0.3.")
t0 = time()
Results["momentum 0.3"] = train(train_loader, data_set, model, criterion, optimizer, epochs=epochs)
print("The whole training process, which consists of", epochs, "epochs, took", round(time()-t0, 2), "seconds.")
print("Maximum accuracy value achieved is: ", round(Results["momentum 0.3"]['Accuracy'][-1], 1), "%")
plot_decision_regions_3class(data_set, model)
plot_accuracy_loss(Results["momentum 0.3"])

# Create a network to classify three classes with 1 hidden layer with 50 neurons and a momentum value of 0.4.
model = Net(Layers)
optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.4)
print("\nLets train the Neural Network through optimizer with momentum=0.4.")
t0 = time()
Results["momentum 0.4"] = train(train_loader, data_set, model, criterion, optimizer, epochs=epochs)
print("The whole training process, which consists of", epochs, "epochs, took", round(time()-t0, 2), "seconds.")
print("Maximum accuracy value achieved is: ", round(Results["momentum 0.4"]['Accuracy'][-1], 1), "%")
plot_decision_regions_3class(data_set, model)
plot_accuracy_loss(Results["momentum 0.4"])

# Create a network to classify three classes with 1 hidden layer with 50 neurons and a momentum value of 0.5.
model = Net(Layers)
optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.5)
print("\nLets train the Neural Network through optimizer with momentum=0.5.")
t0 = time()
Results["momentum 0.5"] = train(train_loader, data_set, model, criterion, optimizer, epochs=epochs)
print("The whole training process, which consists of", epochs, "epochs, took", round(time()-t0, 2), "seconds.")
print("Maximum accuracy value achieved is: ", round(Results["momentum 0.5"]['Accuracy'][-1], 1), "%")
plot_decision_regions_3class(data_set, model)
plot_accuracy_loss(Results["momentum 0.5"])


# #### Compare Results of Different Momentum Terms

# The plot below compares results of different momentum terms. We see that in general. The Cost decreases proportionally
# to the momentum term, but larger momentum terms lead to larger oscillations. While the momentum term decreases faster,
# it seems that a momentum term of 0.2 reaches the smallest value for the cost.

# Plot the Loss result for each term
plt.figure()
for key, value in Results.items():
    plt.plot(value['Cost'], label=key)
plt.legend()
plt.xlabel('Epoch')
plt.ylabel('Cost')
plt.title('Total-Loss on Training Set at each Epoch')
plt.show()

# Plot the Accuracy result for each term
plt.figure()
for key, value in Results.items():
    plt.plot(value['Accuracy'], label=key)
plt.legend()
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Accuracy on whole DataSet at each Epoch')
plt.show()
# The accuracy seems to be proportional to the momentum term.

# Plot maximum accuracy for all models
moment = [0, 0.1, 0.2, 0.3, 0.4, 0.5]
max_accuracy = [Results[k]['Accuracy'][-1] for k in Results.keys()]
plt.figure()
plt.plot(moment, max_accuracy)
plt.xlabel('Momentum Value')
plt.ylabel('Maximum Accuracy')
plt.title('Maximum accuracy value achieved from each optimizer')
plt.show()
