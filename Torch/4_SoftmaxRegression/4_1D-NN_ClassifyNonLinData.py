#!/usr/bin/env python
# coding: utf-8

# In this lab, you will use a single-layer neural network to classify non linearly separable data in 1-D databases.
# Particularly, we will create and use a first dataset that can be correctly classified using a single neuron in the
# hidden layer, and then we will create and use a similar dataset which instead needs for 3 neurons in the hidden layer
# in order to be correctly classified.
#   - Neural Network Module and Training Function
#   - Make Some Data
#   - Define the Neural Network, Criterion Function, Optimizer, and Train the Model

from time import time
import torch
import torch.nn as nn
import matplotlib.pylab as plt
from torch.utils.data import Dataset, DataLoader

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

# Create the Cross-Entropy loss function.
def criterion_cross(outputs, labels):
    out = -1 * torch.mean(labels * torch.log(outputs) + (1 - labels) * torch.log(1 - outputs))
    return out


# Define the activations and the output of the first linear layer as an attribute. Note that this is not good practice.
class Net(nn.Module):

    # Constructor
    def __init__(self, D_in, H, D_out):
        super(Net, self).__init__()
        # hidden layer 
        self.linear1 = nn.Linear(D_in, H)
        self.linear2 = nn.Linear(H, D_out)
        # Define the first linear layer as an attribute, this is not good practice
        self.a1 = None

    # Prediction
    def forward(self, x):
        self.a1 = torch.sigmoid(self.linear1(x))
        yhat = torch.sigmoid(self.linear2(self.a1))
        return yhat


# Define the training function.
def train(data, model, criterion, optimizer, epochs=5, plot_number=10):
    if type(data) == torch.utils.data.dataloader.DataLoader:
        X, Y = (data.dataset.x, data.dataset.y)
    else:
        X, Y = (data.x, data.y)
    cost = []
    ti = time()
    for epoch in range(epochs):
        tot_loss = 0
        for x, y in data:
            yhat = model(x)
            loss = criterion(yhat, y)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            tot_loss += loss.item()  # cumulative loss
        cost.append(tot_loss)
        if epoch % plot_number == 0:
            print("\nTraining process after", epoch, "epochs took", round(time() - ti, 2), "seconds.")
            PlotStuff(X, Y, model, epoch)
            plt.figure()
            plt.scatter(model.a1.detach().numpy()[:, 0], model.a1.detach().numpy()[:, 1], c=Y.numpy().reshape(-1))
            plt.title('Activations')
            plt.show()
            ti = time()
    return cost


# #### Make Some Data

# Define the class to get our FIRST dataset.
class Data1(Dataset):
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


# Define the class to get our SECOND dataset.
class Data2(Dataset):
    def __init__(self):
        self.x = torch.linspace(-20, 20, 100).view(-1, 1)
        self.y = torch.zeros(self.x.shape[0])
        self.y[(self.x[:, 0] > -10) & (self.x[:, 0] < -5)] = 1
        self.y[(self.x[:, 0] > 5) & (self.x[:, 0] < 10)] = 1
        self.y = self.y.view(-1, 1)
        self.len = self.x.shape[0]

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return self.len


# Make the FIRST dataset
data_set1 = Data1()

# Make the SECOND dataset
data_set2 = Data2()


# #### Define the Neural Network, Criterion Function, Optimizer and Train the Model

## For the FIRST dataset
print('\n--> Working on the FIRST dataset.')

# Plot the data
PlotStuff(data_set1.x, data_set1.y)

# Train the model
D_in = 1  # size of input
H1 = 2  # size of hidden layer
D_out = 1  # number of outputs
lr = 0.1  # learning rate
epochs1 = 1000  # number of epochs for iteration
model1 = Net(D_in, H1, D_out)  # create the model
criterion_bce = nn.BCELoss()  # binary cross entropy loss criterion; equally we can do: criterion_bce = criterion_cross
optimizer1 = torch.optim.SGD(model1.parameters(), lr=lr)  # optimizer
t0 = time()
print("\nLets train the model with Binary Cross-Entropy (BCE) loss criterion:")
cost_bce1 = train(data_set1, model1, criterion_bce, optimizer1, epochs=epochs1, plot_number=300)  # train the model
print("\nThe whole training process, which consists of", epochs1, "epochs, took", round(time()-t0, 2), "seconds.")

# Plot the cost
print("\nThe minimum BCE loss achieved after", epochs1, "epochs is", round(min(cost_bce1), 2))
plt.figure()
plt.plot(cost_bce1)
plt.xlabel('Epoch')
plt.ylabel('Cost')
plt.title('Cross-Entropy Loss')
plt.show()
# By examining the output of the activation, you see by the 600th epoch that the data has been mapped to a linearly
# separable space.

# Lets re-train the model but now with MSE Loss Function
criterion_mse = nn.MSELoss()
model1 = Net(D_in, H1, D_out)
optimizer1 = torch.optim.SGD(model1.parameters(), lr=lr)
t0 = time()
print("\n\nLets train the model with Mean Squared Error (MSE) loss criterion:")
cost_mse1 = train(data_set1, model1, criterion_mse, optimizer1, epochs=epochs1, plot_number=300)
print("\nThe whole training process, which consists of", epochs1, "epochs, took", round(time()-t0, 2), "seconds.")

# Plot the cost
print("\nThe minimum MSE loss achieved after", epochs1, "epochs is", round(min(cost_mse1), 2))
plt.figure()
plt.plot(cost_bce1, label='BCE')
plt.plot(cost_mse1, label='MSE')
plt.legend()
plt.xlabel('Epoch')
plt.ylabel('Cost')
plt.title('MSE vs BCE Loss')
plt.show()

## For the SECOND dataset (Note: we use Adam optimizer in this case as it guaranties better performances!)
print('\n\n--> Working on the SECOND dataset.')

# Plot the data
PlotStuff(data_set2.x, data_set2.y)

# Train the model
trainset_data2 = DataLoader(dataset=data_set2, batch_size=100)  # create training set
H2 = 9  # size of hidden layer
epochs2 = 600  # number of epochs for iteration
model2 = Net(D_in, H2, D_out)  # create the model
optimizer2 = torch.optim.Adam(model2.parameters(), lr=lr)  # Adam optimizer
t0 = time()
cost_bce2 = train(trainset_data2, model2, criterion_bce, optimizer2, epochs=epochs2, plot_number=200)  # train the model
print("\nThe whole training process, which consists of", epochs2, "epochs, took", round(time()-t0, 2), "seconds.")

# Plot the loss
print("\nThe minimum BCE loss achieved after", epochs2, "epochs is", round(min(cost_bce2), 2))
plt.figure()
plt.plot(cost_bce2)
plt.xlabel('Epoch')
plt.ylabel('Cost')
plt.title('Cross-Entropy Loss')
plt.show()
