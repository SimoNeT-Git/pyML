#!/usr/bin/env python
# coding: utf-8

# In this lab, you will create a model the PyTorch way. This will help you more complicated models.
#  - Make Some Data
#  - Create the Model and Cost Function the PyTorch way
#  - Train the Model: Batch Gradient Descent

from torch import nn, optim
import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from torch.utils.data import Dataset, DataLoader

# Set the random seed:
torch.manual_seed(1)


# The function for plotting 2D
def Plot_2D_Plane(model, data, n=0):
    w1 = model.state_dict()['linear.weight'].numpy()[0][0]
    w2 = model.state_dict()['linear.weight'].numpy()[0][1]
    b = model.state_dict()['linear.bias'].numpy()

    # Data
    x1 = data.x[:, 0].view(-1, 1).numpy()
    x2 = data.x[:, 1].view(-1, 1).numpy()
    y = data.y.numpy()

    # Make plane
    X1, X2 = np.meshgrid(np.arange(x1.min(), x1.max(), 0.05), np.arange(x2.min(), x2.max(), 0.05))
    yhat = w1 * X1 + w2 * X2 + b

    # Plotting
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot(x1[:, 0], x2[:, 0], y[:, 0], 'ro', label='Data Points')  # Scatter plot
    ax.plot_surface(X1, X2, yhat)  # Plane plot
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.set_zlabel('y')
    plt.title('Estimated Plane at Iteration ' + str(n))
    ax.legend()
    plt.show()


# #### Make Some Data

# Create a dataset class with two-dimensional features:
class Data2D(Dataset):

    # Constructor
    def __init__(self):
        self.x = torch.zeros(20, 2)
        self.x[:, 0] = torch.arange(-1, 1, 0.1)
        self.x[:, 1] = torch.arange(-1, 1, 0.1)
        self.w = torch.tensor([[1.0], [1.0]])
        self.b = 1
        self.f = torch.mm(self.x, self.w) + self.b
        self.y = self.f + 0.1 * torch.randn((self.x.shape[0], 1))
        self.len = self.x.shape[0]

    # Getter
    def __getitem__(self, index):
        return self.x[index], self.y[index]

    # Get Length
    def __len__(self):
        return self.len


# Create a dataset object:
data_set = Data2D()

# Create a data loader object. Set the batch_size equal to 2:
batch = 2
train_loader = DataLoader(dataset=data_set, batch_size=batch)


# #### Create the Model, Optimizer, and Total Loss Function (Cost)

# Create a customized linear regression module:
class linear_regression(nn.Module):

    # Constructor
    def __init__(self, input_size, output_size):
        super(linear_regression, self).__init__()
        self.linear = nn.Linear(input_size, output_size)

    # Prediction
    def forward(self, x):
        yhat = self.linear(x)
        return yhat


# Create the linear regression model and print the parameters
model = linear_regression(2, 1)
params = list(model.parameters())
print("\n\nThe weights of the custom linear_regression class that uses the torch.nn.linear model "
      "randomly initialized are", params[0].tolist(), "\nwith shape", params[0].shape[0], "x", params[0].shape[1],
      "while the bias is", params[1].tolist())

# Create an optimizer object. Set the learning rate to 0.1 and enter the model parameters in the constructor.
lr = 0.1
optimizer = optim.SGD(model.parameters(), lr=lr)

# Create the criterion function that calculates the total loss or cost:
criterion = nn.MSELoss()


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
# is an approximation of the true total loss or cost:
print("\nTraining the model with a batch_size of 2...")
Plot_2D_Plane(model, data_set)
epochs = 100
LOSS, COST = train_model(train_loader, epochs, model, optimizer, criterion)
Plot_2D_Plane(model, data_set, epochs)

# Create a new model1. Train the model with a batch size 30 and learning rate 0.1, store the loss or total cost in a
# list LOSS1, and plot the results.
data_set1 = Data2D()
batch1 = 10
train_loader1 = DataLoader(dataset=data_set1, batch_size=batch1)
model1 = linear_regression(2, 1)
optimizer1 = optim.SGD(model1.parameters(), lr=lr)

# Train the model
print("\nTraining the model with a batch_size of 30...")
Plot_2D_Plane(model1, data_set1)
LOSS1, COST1 = train_model(train_loader1, epochs, model1, optimizer1, criterion)
Plot_2D_Plane(model1, data_set1, epochs)

# Plot out the Loss and iteration diagram
plt.figure(figsize=(12, 6))
plt.subplot(121)
plt.plot(LOSS, label='Batch Size = ' + str(batch))
plt.plot(LOSS1, label='Batch Size = ' + str(batch1))
plt.legend()
plt.xlabel("Iterations\nTotal number of iterations = (number of epochs)x(data length)/(batch size) = "
           + str(epochs) + "x" + str(len(data_set)) + "/" + str(batch) + " = [" + str(epochs*len(data_set)/batch) +
           " and " + str(epochs*len(data_set1)/batch1) + "]")
plt.ylabel("Loss")
plt.title('Loss value at each iteration')
plt.subplot(122)
plt.plot(COST, label='Batch Size = ' + str(batch))
plt.plot(COST1, label='Batch Size = ' + str(batch1))
plt.legend()
plt.xlabel("Epochs")
plt.ylabel("Cost/Total-Loss")
plt.title('Total-Loss at each epoch')
plt.show()

# Use the following validation data to calculate the total loss or cost for both models:
torch.manual_seed(2)
validation_data = Data2D()
Y_val = validation_data.y
X_val = validation_data.x
print("\nTotal loss or cost for the first model (batch size = " + str(batch) + "): ",
      criterion(model(X_val), Y_val).tolist())
print("Total loss or cost for the second model (batch size = " + str(batch1) + "): ",
      criterion(model1(X_val), Y_val).tolist())
