#!/usr/bin/env python
# coding: utf-8

# In this lab, you will see how many neurons it takes to classify noisy XOR data with one hidden layer neural network.
#   - Neural Network Module and Training Function
#   - Make Some Data
#   - One Neuron
#   - Two Neurons
#   - Three Neurons

from time import time
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from torch.utils.data import Dataset, DataLoader

torch.manual_seed(0)


# Use the following function to plot the data:
def plot_decision_regions_2class(data_set, model):
    cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#00AAFF'])
    # cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#00AAFF'])
    X = data_set.x.numpy()
    y = data_set.y.numpy()
    h = .02
    x_min, x_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
    y_min, y_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    XX = torch.Tensor(np.c_[xx.ravel(), yy.ravel()])
    yhat = np.logical_not((model(XX)[:, 0] > 0.5).numpy()).reshape(xx.shape)

    plt.figure()
    plt.pcolormesh(xx, yy, yhat, cmap=cmap_light)
    plt.plot(X[y[:, 0] == 0, 0], X[y[:, 0] == 0, 1], 'o', label='y=0')
    plt.plot(X[y[:, 0] == 1, 0], X[y[:, 0] == 1, 1], 'ro', label='y=1')
    plt.title("Decision Region")
    plt.legend()
    plt.show()


# Use the following function to show cost and accuracy of a model on the same plot
def plot_accloss(LOSS, ACC):
    fig, ax1 = plt.subplots(figsize=(8, 6))
    color = 'tab:red'
    ax1.plot(LOSS, color=color)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Total Loss', color=color)
    ax1.tick_params(axis='y', color=color)

    ax2 = ax1.twinx()
    color = 'tab:blue'
    ax2.set_ylabel('Accuracy', color=color)
    ax2.plot(ACC, color=color)
    ax2.tick_params(axis='y', color=color)
    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.title('Cost / Accuracy at each Epoch')
    plt.show()


# Use the following function to calculate accuracy:
def accuracy(model, train_loader):
    X, Y = zip(*train_loader)
    X, Y = (torch.stack(X).view(-1, 2), torch.stack(Y).view(-1, 2))
    return np.mean(Y.view(-1).numpy() == (model(X)[:, 0] > 0.5).numpy()) * 100


# #### Neural Network Module and Training Function

# Define the neural network (with one hidden layer) module or class:
class Net(nn.Module):

    # Constructor
    def __init__(self, D_in, H, D_out):
        super(Net, self).__init__()
        # hidden layer
        self.linear1 = nn.Linear(D_in, H)
        # output layer
        self.linear2 = nn.Linear(H, D_out)

    # Prediction    
    def forward(self, x):
        x = torch.sigmoid(self.linear1(x))
        x = torch.sigmoid(self.linear2(x))
        return x


# Define a function to train the model:
def train(train_loader, model, criterion, optimizer, epochs=5):
    LOSS = []
    ACC = []
    for epoch in range(epochs):
        total = 0
        for x, y in train_loader:
            optimizer.zero_grad()
            yhat = model(x)
            loss = criterion(yhat, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += loss.item()  # cumulative loss
        ACC.append(accuracy(model, train_loader))
        LOSS.append(total / len(train_loader))

    return LOSS, ACC


# #### Make Some Data

# Define the class XOR_Data
class XOR_Data(Dataset):

    # Constructor
    def __init__(self, N_s=100):
        self.x = torch.zeros((N_s, 2))
        self.y = torch.zeros((N_s, 1))
        for i in range(N_s // 4):
            self.x[i, :] = torch.Tensor([0.0, 0.0])
            self.y[i, 0] = torch.Tensor([0.0])

            self.x[i + N_s // 4, :] = torch.Tensor([0.0, 1.0])
            self.y[i + N_s // 4, 0] = torch.Tensor([1.0])

            self.x[i + N_s // 2, :] = torch.Tensor([1.0, 0.0])
            self.y[i + N_s // 2, 0] = torch.Tensor([1.0])

            self.x[i + 3 * N_s // 4, :] = torch.Tensor([1.0, 1.0])
            self.y[i + 3 * N_s // 4, 0] = torch.Tensor([0.0])

            self.x = self.x + 0.01 * torch.randn((N_s, 2))
        self.len = N_s

    # Getter
    def __getitem__(self, index):
        return self.x[index], self.y[index]

    # Get Length
    def __len__(self):
        return self.len

    # Plot the data
    def plot_stuff(self):
        plt.figure()
        plt.plot(self.x[self.y[:, 0] == 0, 0].numpy(), self.x[self.y[:, 0] == 0, 1].numpy(), 'bo', label="y=0")
        plt.plot(self.x[self.y[:, 0] == 1, 0].numpy(), self.x[self.y[:, 0] == 1, 1].numpy(), 'ro', label="y=1")
        plt.title('XOR data')
        plt.legend()
        plt.show()


# Create dataset object:
data_set = XOR_Data()
data_set.plot_stuff()

# #### 1 Neuron

# Create a neural network model with one neuron.
n1 = 1
model1 = Net(2, n1, 1)

# Define training set and criterion
train_loader = DataLoader(dataset=data_set, batch_size=1)
criterion = nn.BCELoss()

# Train the model
lr1 = 0.001
optimizer1 = torch.optim.SGD(model1.parameters(), lr=lr1)
epochs = 500
t0 = time()
print('\nLets train a Neural Network with', n1, 'neurons in the hidden layer and a learning rate of', lr1,
      'on the XOR dataset...')
LOSS1, ACC1 = train(train_loader, model1, criterion, optimizer1, epochs=epochs)
print("The whole training process, which consists of", epochs, "epochs, took", round(time()-t0, 2), "seconds.")
plot_accloss(LOSS1, ACC1)
plot_decision_regions_2class(data_set, model1)


# #### 2 Neurons

# Create a neural network model with two neurons.
n2 = 2
model2 = Net(2, n2, 1)

# Train the model
lr2 = 0.1
optimizer2 = torch.optim.SGD(model2.parameters(), lr=lr2)
t0 = time()
print('\nLets train a Neural Network with', n2, 'neurons in the hidden layer and a learning rate of', lr2,
      'on the XOR dataset...')
LOSS2, ACC2 = train(train_loader, model2, criterion, optimizer2, epochs=epochs)
print("The whole training process, which consists of", epochs, "epochs, took", round(time()-t0, 2), "seconds.")
plot_accloss(LOSS2, ACC2)
plot_decision_regions_2class(data_set, model2)


# #### 3 Neurons

# Create a neural network model with three neurons.
n3 = 3
model3 = Net(2, n3, 1)

# Train the model
lr3 = 0.1
optimizer3 = torch.optim.SGD(model3.parameters(), lr=lr3)
t0 = time()
print('\nLets train a Neural Network with', n3, 'neurons in the hidden layer and a learning rate of', lr3,
      'on the XOR dataset...')
LOSS3, ACC3 = train(train_loader, model3, criterion, optimizer3, epochs=epochs)
print("The whole training process, which consists of", epochs, "epochs, took", round(time()-t0, 2), "seconds.")
plot_accloss(LOSS3, ACC3)
plot_decision_regions_2class(data_set, model3)

# Compare models' performances
fig = plt.figure(figsize=(8, 5))
plt.subplot(121)
plt.plot(ACC1, label=str(n1) + ' hidden neurons')
plt.plot(ACC2, label=str(n2) + ' hidden neurons')
plt.plot(ACC3, label=str(n3) + ' hidden neurons')
plt.legend()
plt.xlabel('Epochs')
plt.ylabel('Accuracy (%)')
plt.title('Comparing Accuracy', fontsize=10)
plt.subplot(122)
plt.plot(LOSS1, label=str(n1) + ' hidden neurons')
plt.plot(LOSS2, label=str(n2) + ' hidden neurons')
plt.plot(LOSS3, label=str(n3) + ' hidden neurons')
plt.legend()
plt.xlabel('Epochs')
plt.ylabel('Cost')
plt.title('Comparing Total Loss', fontsize=10)
fig.suptitle("Comparison of Neural Networks' performances on classification of XOR data")
plt.show()
