#!/usr/bin/env python
# coding: utf-8

# In this lab, we will see the problem of initializing the weights with the same value. We will see that even for a
# simple network, our model will not train properly.
#   - Neural Network Module and Training Function
#   - Make Some Data
#   - Define the Neural Network with Same Weights Initialization, Criterion Function, Optimizer, and Train the Model
#   - Define the Neural Network with default Weights Initialization, Criterion Function, Optimizer, and Train the Model

from time import time
import torch
import torch.nn as nn
from torch import sigmoid
import matplotlib.pylab as plt
from torch.utils.data import Dataset
torch.manual_seed(0)


# The function for plotting the model.
def PlotStuff(X, Y, model=None, iter=0):
    plt.figure()
    plt.plot(X[Y == 0].numpy(), Y[Y == 0].numpy(), 'or', label='Training Points y=0')
    plt.plot(X[Y == 1].numpy(), Y[Y == 1].numpy(), 'ob', label='Training Points y=1')
    plt.xlabel('x')
    if model is None:
        plt.title('Data Points')
    else:
        plt.plot(X.numpy(), model(X).detach().numpy(), label='Neural Network')
        plt.title('Training at Epoch ' + str(iter))
    plt.legend()
    plt.show()


# #### Neural Network Module and Training Function

# Define the activations and the output of the first linear layer as an attribute. Note that this is not good practice.

# Create the Cross-Entropy loss function:
def criterion_cross(outputs, labels):
    out = -1 * torch.mean(labels * torch.log(outputs) + (1 - labels) * torch.log(1 - outputs))
    return out


# Define the class Net
class Net(nn.Module):

    # Constructor
    def __init__(self, D_in, H, D_out):
        super(Net, self).__init__()
        # hidden layer 
        self.linear1 = nn.Linear(D_in, H)
        self.linear2 = nn.Linear(H, D_out)
        # Define the first linear layer as an attribute, this is not good practice
        self.a1 = None
        self.l1 = None
        self.l2 = None

    # Prediction
    def forward(self, x):
        self.l1 = self.linear1(x)
        self.a1 = sigmoid(self.l1)
        self.l2 = self.linear2(self.a1)
        yhat = sigmoid(self.l2)
        return yhat


# Define the training function:
def train(data, model, optimizer, criterion, epochs=1000):
    X, Y = (data.x, data.y)
    cost = []
    for epoch in range(epochs):
        total = 0
        for x, y in zip(X, Y):
            yhat = model(x)
            loss = criterion(yhat, y)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            # cumulative loss
            total += loss.item()
        cost.append(total / len(X))
        if epoch % 300 == 0:
            PlotStuff(X, Y, model, epoch)
            model(X)
            plt.figure()
            plt.scatter(model.a1.detach().numpy()[:, 0], model.a1.detach().numpy()[:, 1], c=Y.numpy().reshape(-1))
            plt.title('Activations')
            plt.show()
    return cost


# #### Make Some Data

# Define the class to get our dataset.
class Data(Dataset):

    def __init__(self):
        self.x = torch.linspace(-20, 20, 40 + 1).view(-1, 1)
        # the previous line is equal to self.x = torch.arange(-20, 20, 1).view(-1, 1).type(torch.FloatTensor)
        self.y = torch.zeros(self.x.shape[0])
        self.y[(self.x[:, 0] > -4) & (self.x[:, 0] < 4)] = 1.0
        self.y = self.y.view(-1, 1)
        self.len = self.x.shape[0]

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return self.len


# Make some data
data_set = Data()

# Plot the data
PlotStuff(data_set.x, data_set.y)


# #### Define the Neural Network with Same Weights Initialization, Criterion Function, Optimizer and Train the Model

# Define common parameters
D_in = 1  # size of input
H = 2  # size of hidden layer
D_out = 1  # number of outputs
learning_rate = 0.1  # learning rate
epochs = 1000

# Create the model
print("\nThe first neural network is trained with uniform initial parameters.")
model1 = Net(D_in, H, D_out)

# Same Weights Initialization with all ones for weights and zeros for the bias.
model1.state_dict()['linear1.weight'][0] = 1.0
model1.state_dict()['linear1.weight'][1] = 1.0
model1.state_dict()['linear1.bias'][0] = 0.0
model1.state_dict()['linear1.bias'][1] = 0.0
model1.state_dict()['linear2.weight'][0] = 1.0
model1.state_dict()['linear2.bias'][0] = 0.0
print("Custom made uniform parameters for the model are:")
for k, v in zip(list(model1.state_dict().keys()), list(model1.state_dict().values())):
    print(k, ":  ", v.tolist())

# Optimizer and Criterion
criterion_bce = nn.BCELoss()
optimizer1 = torch.optim.SGD(model1.parameters(), lr=learning_rate)

# Train the Model
t0 = time()
cost_cross1 = train(data_set, model1, optimizer1, criterion_bce, epochs=epochs)
print("The whole training process, which consists of", epochs, "epochs, took", round(time()-t0, 2), "seconds.")

# By examining the output of the parameters, although they have changed from before they are identical between them.
print("After training the model, the parameters are:")
for k, v in zip(list(model1.state_dict().keys()), list(model1.state_dict().values())):
    print(k, ":  ", v.tolist())
yhat1 = model1(data_set.x)
PlotStuff(data_set.x, data_set.y, model=model1, iter=epochs)


# #### Define the Neural Network with default initialization, Optimizer and Train the Model

# Define the Neural Network
print("\nThe second neural network is trained with default random initial parameters.")
model2 = Net(D_in, H, D_out)  # create the model

# This is the PyTorch default initialization
print("Default initial parameters of the model are:")
for k, v in zip(list(model2.state_dict().keys()), list(model2.state_dict().values())):
    print(k, ":  ", v.tolist())

# Optimizer and Criterion
optimizer2 = torch.optim.SGD(model2.parameters(), lr=learning_rate)

# Train the model
t0 = time()
cost_cross2 = train(data_set, model2, optimizer2, criterion_bce, epochs=epochs)
print("The whole training process, which consists of", epochs, "epochs, took", round(time()-t0, 2), "seconds.")
print("After training the model, the parameters are:")
for k, v in zip(list(model2.state_dict().keys()), list(model2.state_dict().values())):
    print(k, ":  ", v.tolist())
yhat2 = model2(data_set.x)
PlotStuff(data_set.x, data_set.y, model=model2, iter=epochs)

# plot the loss
fig = plt.figure()
plt.plot(cost_cross1, 'r', label='Uniform Initialization')
plt.plot(cost_cross2, 'b', label='Default Initialization')
plt.legend()
plt.xlabel('Epoch')
plt.ylabel('Cost')
plt.title('Comparison of Cross-Entropy on Training Data')
plt.show()

# Plot predictions and real data
plt.figure()
plt.plot(data_set.x, data_set.y, 'go', label='Data Points')
plt.plot(data_set.x, yhat1.detach().numpy(), 'r', label='Uniform Initialization')
plt.plot(data_set.x, yhat2.detach().numpy(), 'b', label='Default Initialization')
plt.legend()
plt.xlabel('x')
plt.ylabel('y')
plt.title('Real Data and Predictions')
plt.show()
