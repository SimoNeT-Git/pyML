#!/usr/bin/env python
# coding: utf-8

# In this lab, you will create a model the Pytroch way. This will help you as models get more complicated.
#  - Make Some Data
#  - Create the Model and Cost Function the Pytorch way
#  - Train the Model: Batch Gradient Descent

import torch
import numpy as np
import matplotlib.pyplot as plt
from torch import nn, optim
from mpl_toolkits.mplot3d import Axes3D
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

# Set the random seed:
torch.manual_seed(1)

# #### Make Some Data

# Create a dataset class with two-dimensional features and two targets:
from torch.utils.data import Dataset, DataLoader


class Data(Dataset):
    def __init__(self):
        self.x = torch.zeros(20, 2)
        self.x[:, 0] = torch.arange(-1, 1, 0.1)
        self.x[:, 1] = torch.arange(-1, 1, 0.1)
        self.w = torch.tensor([[1.0, -1.0], [1.0, 3.0]])
        self.b = torch.tensor([[1.0, -1.0]])
        self.f = torch.mm(self.x, self.w) + self.b
        self.y = self.f + 0.001 * torch.randn((self.x.shape[0], 1))
        self.len = self.x.shape[0]

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return self.len


# create a dataset object
data_set = Data()


# #### Create the Model, Optimizer, and Total Loss Function (cost)

# Create a custom module:
class linear_regression(nn.Module):

    def __init__(self, input_size, output_size):
        super(linear_regression, self).__init__()
        self.linear = nn.Linear(input_size, output_size)

    def forward(self, x):
        yhat = self.linear(x)
        return yhat


# Create an optimizer object and set the learning rate to 0.1. Don't forget to enter model parameters in the constructor
model = linear_regression(2, 2)

# Create an optimizer object and set the learning rate to 0.1. Don't forget to enter model parameters in the constructor
optimizer = optim.SGD(model.parameters(), lr=0.1)

# Create the criterion function that calculates the total loss or cost:
criterion = nn.MSELoss()

# Create a data loader object and set the batch_size to 5:
batch = 5
train_loader = DataLoader(dataset=data_set, batch_size=batch)


# #### Train the Model via Mini-Batch Gradient Descent
def train_model(data, n_iters, modello, optim, criterio):
    # X_tr, Y_tr = zip(*data)
    # X_tr, Y_tr = (torch.stack(X_tr).view(-1, 1), torch.stack(Y_tr).view(-1, 1))
    loss_list = []
    cost_list = []
    for epoch in range(n_iters):
        tot_loss = 0
        for x, y in data:
            yhat = modello(x)
            loss = criterio(yhat, y)
            loss_list.append(loss.item())
            tot_loss += loss.item()
            optim.zero_grad()
            loss.backward()
            optim.step()
        cost_list.append(tot_loss / len(data))
    return loss_list, cost_list


# Run 100 epochs of Mini-Batch Gradient Descent and store the total loss or cost for every iteration. Remember that this
# is an approximation of the true total loss or cost.
epochs = 100
LOSS, COST = train_model(train_loader, epochs, model, optimizer, criterion)

# Plot the cost:
# Plot out the Loss and iteration diagram
plt.figure(figsize=(12, 6))
plt.subplot(121)
plt.plot(LOSS)
plt.xlabel("Iterations\nTotal number of iterations = (number of epochs)x(data length)/(batch size) = "
           + str(epochs) + "x" + str(len(data_set)) + "/" + str(batch) + " = " + str(epochs*len(data_set)/batch))
plt.ylabel("Loss")
plt.title("Loss value at each iteration")
plt.subplot(122)
plt.plot(COST)
plt.xlabel("Epochs")
plt.ylabel("Cost/Total-Loss")
plt.title('Total-Loss at each epoch')
plt.show()
