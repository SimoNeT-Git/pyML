#!/usr/bin/env python
# coding: utf-8

# In this Lab, you will practice training a model by using Mini-Batch Gradient Descent.
#  - Make Some Data
#  - Create the Model and Cost Function (Total Loss)
#  - Train the Model: Batch Gradient Descent
#  - Train the Model: Stochastic Gradient Descent with Dataset DataLoader
#  - Train the Model: Mini Batch Gradient Decent: Batch Size Equals 5
#  - Train the Model: Mini Batch Gradient Decent: Batch Size Equals 10

import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d


# The class plot_error_surfaces is just to help you visualize the data space and the parameter space during training and
# has nothing to do with PyTorch.
class plot_error_surfaces(object):

    # Constructor
    def __init__(self, w_range, b_range, X, Y, n_samples=30, go=True):
        W = np.linspace(-w_range, w_range, n_samples)
        B = np.linspace(-b_range, b_range, n_samples)
        w, b = np.meshgrid(W, B)
        Z = np.zeros((30, 30))
        count1 = 0
        self.y = Y.numpy()
        self.x = X.numpy()
        for w1, b1 in zip(w, b):
            count2 = 0
            for w2, b2 in zip(w1, b1):
                Z[count1, count2] = np.mean((self.y - w2 * self.x + b2) ** 2)
                count2 += 1
            count1 += 1
        self.Z = Z
        self.w = w
        self.b = b
        self.W = []
        self.B = []
        self.LOSS = []
        self.n = 0
        if go:
            # 3D visualization (surface) of loss with respect to weight and bias
            plt.figure(figsize=(7.5, 5))
            ax = plt.axes(projection='3d')
            ax.plot_surface(self.w, self.b, self.Z,
                            rstride=1, cstride=1, cmap='viridis', edgecolor='none')
            ax.set_title('Cost/Total-Loss Surface')
            ax.set_xlabel('Weight')
            ax.set_ylabel('Bias')
            ax.set_zlabel('Cost')
            plt.show()
            # 2D visualization (surface contour) of loss with respect to weight and bias
            plt.figure()
            plt.title('Cost/Total-Loss Surface Contour')
            plt.xlabel('Weight')
            plt.ylabel('Bias')
            plt.contourf(self.w, self.b, self.Z)
            plt.colorbar()
            plt.show()

    # Setter
    def set_para_loss(self, W, B, loss):
        self.n = self.n + 1
        self.W.append(W)
        self.B.append(B)
        self.LOSS.append(loss)

    # Plot diagram during iterations
    def plot_ps(self):
        plt.figure(figsize=(9, 5))
        plt.subplot(121)
        plt.plot(self.x, self.y, 'bo', label="Data Points")
        plt.plot(self.x, self.W[-1] * self.x + self.B[-1], 'r', label="Regression")
        plt.xlabel('x')
        plt.ylabel('y')
        plt.ylim((-10, 15))
        plt.title('Data Space Iteration: ' + str(self.n))
        plt.subplot(122)
        plt.contour(self.w, self.b, self.Z)
        plt.scatter(self.W, self.B, c='r', marker='x')
        plt.title('Total-Loss Surface Contour Iteration ' + str(self.n))
        plt.xlabel('Weight')
        plt.ylabel('Bias')
        plt.show()

    # Final diagram
    def final_plot(self):
        # 3D visualization (surface)
        plt.figure()
        ax = plt.axes(projection='3d')
        ax.plot_wireframe(self.w, self.b, self.Z)
        ax.scatter(self.W, self.B, self.LOSS, c='r', marker='x', s=200, alpha=1)
        ax.set_xlabel('Weight')
        ax.set_ylabel('Bias')
        ax.set_zlabel('Cost')
        ax.set_title('Final Cost/Total-Loss Surface')
        plt.show()
        # 2D visualization (surface contour)
        plt.figure()
        plt.contour(self.w, self.b, self.Z)
        plt.scatter(self.W, self.B, c='r', marker='x')
        plt.xlabel('Weight')
        plt.ylabel('Bias')
        plt.title('Final Cost/Total-Loss Surface Contour')
        plt.show()


# #### Make Some Data

# Set random seed:
torch.manual_seed(1)

# Generate values from -3 to 3 that create a line with a slope of 1 and a bias of -1. This is the line that you need to
# estimate. Add some noise to the data:
X = torch.arange(-3, 3, 0.1).view(-1, 1)
f = 1 * X - 1
Y = f + 0.1 * torch.randn(X.size())

# Plot out the data dots and line
plt.figure()
plt.plot(X.numpy(), Y.numpy(), 'rx', label='Data Points')
plt.plot(X.numpy(), f.numpy(), label='Fit')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()


# #### Create the Model and Cost Function (Total Loss)

# Define the prediction function
def forward(x):
    return w * x + b


# Define the cost or criterion function:
def criterion(yhat, y):
    return torch.mean((yhat - y) ** 2)


# Create a plot_error_surfaces object to visualize the data space and the parameter space during training:
get_surface_GD = plot_error_surfaces(15, 13, X, Y, 30)

# #### Train the Model: Batch Gradient Descent (BGD)

# Define train_model_BGD function.
w0 = -15.0
b0 = -10.0
w = torch.tensor(w0, requires_grad=True)
b = torch.tensor(b0, requires_grad=True)
lr = 0.1


def train_model(Xvar, Yvar, n_iter, weight, bias, eta, surf):
    # Create empty list to save loss values at each iteration:
    loss_list = []
    for epoch in range(n_iter):
        # make a prediction
        Yhat = forward(Xvar)
        # calculate the loss
        loss = criterion(Yhat, Yvar)
        # store the loss in the list LOSS_BGD
        loss_list.append(loss)

        # Section for plotting
        surf.set_para_loss(weight.data.tolist(), bias.data.tolist(), loss.tolist())
        surf.plot_ps()

        # backward pass: compute gradient of the loss with respect to all the learnable parameters
        loss.backward()
        # update parameters slope and bias
        weight.data = weight.data - eta * weight.grad.data
        bias.data = bias.data - eta * bias.grad.data
        # zero the gradients before running the backward pass
        weight.grad.data.zero_()
        bias.grad.data.zero_()

    return loss_list


# Run 10 epochs of batch gradient descent: bug data space is 1 iteration ahead of parameter space.
LOSS_GD = train_model(X, Y, 10, w, b, lr, get_surface_GD)

# #### Stochastic Gradient Descent (SGD) with Dataset DataLoader
from torch.utils.data import Dataset, DataLoader


# Create class Data
class Data(Dataset):

    # Constructor
    def __init__(self):
        self.x = torch.arange(-3, 3, 0.1).view(-1, 1)
        self.y = 1 * X - 1
        self.len = self.x.shape[0]

    # Getter
    def __getitem__(self, index):
        return self.x[index], self.y[index]

    # Get length
    def __len__(self):
        return self.len


# Create Data object and DataLoader object
dataset = Data()
trainloader = DataLoader(dataset=dataset)  # batch size is 1 by default

# Create a plot_error_surfaces object to visualize the data space and the parameter space during training:
get_surface_SGD = plot_error_surfaces(15, 13, dataset.x, dataset.y, 30, go=False)

# Define train_model_SGD function for training the model.
w = torch.tensor(w0, requires_grad=True)
b = torch.tensor(b0, requires_grad=True)


def train_model_SGD(XY, n_iter, weight, bias, eta, surf):
    Xvar, Yvar = zip(*XY)
    Xvar, Yvar = (torch.stack(Xvar).view(-1, 1), torch.stack(Xvar).view(-1, 1))

    loss_list = []
    cost_list = []

    for epoch in range(n_iter):
        # SGD is an approximation of out true total-loss/cost, in this line of code we calculate our true loss/cost and
        # store it
        Yhat = forward(Xvar)
        # store the loss. This is an approximation of total lost of SGD method, the actual one is stored in cost_list
        loss_list.append(criterion(Yhat, Yvar).tolist())
        tot_loss = 0

        for x, y in XY:
            # make a prediction
            yhat = forward(x)
            # calculate the loss
            loss = criterion(yhat, y)
            # update total loss summing the loss value for each data point
            tot_loss += loss.item()

            # Section for plotting
            surf.set_para_loss(weight.data.tolist(), bias.data.tolist(), loss.tolist())

            # backward pass: compute gradient of the loss with respect to all the learnable parameters
            loss.backward()
            # update parameters slope and bias
            weight.data = weight.data - eta * weight.grad.data
            bias.data = bias.data - eta * bias.grad.data
            # zero the gradients before running the backward pass
            weight.grad.data.zero_()
            bias.grad.data.zero_()

        # store the mean (over the dataset) of the total loss (computed at each epoch on the whole dataset)
        cost_list.append(tot_loss/len(XY))

        # plot surface and data space after each epoch
        surf.plot_ps()

    return loss_list, cost_list


# Run 10 epochs of stochastic gradient descent: bug data space is 1 iteration ahead of parameter space.
LOSS_SGD, cost_SGD = train_model_SGD(trainloader, 10, w, b, lr, get_surface_SGD)

# Plot the loss for each epoch:
plt.figure()
plt.plot(LOSS_GD, label="Batch Gradient Descent")
plt.plot(cost_SGD, label="Stochastic Gradient Descent")
# plt.plot(LOSS_SGD, label="Stochastic Gradient Descent")
plt.legend()
plt.tight_layout()
plt.xlabel("Epoch/Iterations")
plt.ylabel("Cost")
plt.title('Loss value at each iteration')
plt.show()


# #### Mini Batch Gradient Descent: Batch Size Equals 5

# Create Data object and create a Dataloader object where the batch size equals 5:
dataset5 = Data()
trainloader5 = DataLoader(dataset=dataset5, batch_size=5)

# Create a plot_error_surfaces object to visualize the data space and the parameter space during training:
get_surface_MBGD5 = plot_error_surfaces(15, 13, dataset5.x, dataset5.y, 30, go=False)

# Define train_model_Mini5 function to train the model.
w = torch.tensor(-15.0, requires_grad=True)
b = torch.tensor(-10.0, requires_grad=True)


# def train_model_Mini5(epochs):
#     for epoch in range(epochs):
#         Yhat = forward(X)
#         get_surface.set_para_loss(w.data.tolist(), b.data.tolist(), criterion(Yhat, Y).tolist())
#         get_surface.plot_ps()
#         LOSS_MINI5.append(criterion(forward(X), Y).tolist())
#         for x, y in trainloader:
#             yhat = forward(x)
#             loss = criterion(yhat, y)
#             get_surface.set_para_loss(w.data.tolist(), b.data.tolist(), loss.tolist())
#             loss.backward()
#             w.data = w.data - lr * w.grad.data
#             b.data = b.data - lr * b.grad.data
#             w.grad.data.zero_()
#             b.grad.data.zero_()


# Run 10 epochs of mini-batch gradient descent: bug data space is 1 iteration ahead of parameter space.
LOSS_MBGD5, cost_MBGD5 = train_model_SGD(trainloader5, 10, w, b, lr, get_surface_MBGD5)

# #### Mini Batch Gradient Descent: Batch Size Equals 10

# Create Data object and create a Dataloader object batch size equals 10
dataset10 = Data()
trainloader10 = DataLoader(dataset=dataset10, batch_size=10)

# Create a plot_error_surfaces object to visualize the data space and the parameter space during training:
get_surface_MBGD10 = plot_error_surfaces(15, 13, dataset10.x, dataset10.y, 30, go=False)

# Define train_model_Mini10 function for training the model.
w = torch.tensor(w0, requires_grad=True)
b = torch.tensor(b0, requires_grad=True)


# def train_model_Mini10(epochs):
#     for epoch in range(epochs):
#         Yhat = forward(X)
#         get_surface.set_para_loss(w.data.tolist(), b.data.tolist(), criterion(Yhat, Y).tolist())
#         get_surface.plot_ps()
#         LOSS_MINI10.append(criterion(forward(X), Y).tolist())
#         for x, y in trainloader:
#             yhat = forward(x)
#             loss = criterion(yhat, y)
#             get_surface.set_para_loss(w.data.tolist(), b.data.tolist(), loss.tolist())
#             loss.backward()
#             w.data = w.data - lr * w.grad.data
#             b.data = b.data - lr * b.grad.data
#             w.grad.data.zero_()
#             b.grad.data.zero_()


# Run 10 epochs of mini-batch gradient descent: bug data space is 1 iteration ahead of parameter space.
LOSS_MBGD10, cost_MBGD10 = train_model_SGD(trainloader10, 10, w, b, lr, get_surface_MBGD10)

# Plot the loss for each epoch:
plt.figure()
plt.plot(LOSS_GD, label="Batch Gradient Descent")
# plt.plot(LOSS_SGD, label="Stochastic Gradient Descent")
# plt.plot(LOSS_MBGD5, label="Mini-Batch Gradient Descent, Batch size=5")
# plt.plot(LOSS_MBGD10, label="Mini-Batch Gradient Descent, Batch size=10")
plt.plot(cost_SGD, label="Stochastic Gradient Descent")
plt.plot(cost_MBGD5, label="Mini-Batch Gradient Descent, Batch size=5")
plt.plot(cost_MBGD10, label="Mini-Batch Gradient Descent, Batch size=10")
plt.legend()
plt.tight_layout()
plt.xlabel("Epoch/Iterations")
plt.ylabel("Cost")
plt.title('Loss value at each iteration')
plt.show()

# Perform mini batch gradient descent with a batch size of 20. Store the total loss for each epoch in the list LOSS20.
dataset20 = Data()
trainloader20 = DataLoader(dataset=dataset20, batch_size=20)
get_surface_MBGD20 = plot_error_surfaces(15, 13, dataset20.x, dataset20.y, 30, go=False)
w = torch.tensor(w0, requires_grad=True)
b = torch.tensor(b0, requires_grad=True)


# def my_train_model(epochs):
#     for epoch in range(epochs):
#         Yhat = forward(X)
#         get_surface.set_para_loss(w.data.tolist(), b.data.tolist(), criterion(Yhat, Y).tolist())
#         get_surface.plot_ps()
#         LOSS_MINI20.append(criterion(forward(X), Y).tolist())
#         for x, y in trainloader:
#             yhat = forward(x)
#             loss = criterion(yhat, y)
#             get_surface.set_para_loss(w.data.tolist(), b.data.tolist(), loss.tolist())
#             loss.backward()
#             w.data = w.data - lr * w.grad.data
#             b.data = b.data - lr * b.grad.data
#             w.grad.data.zero_()
#             b.grad.data.zero_()


# Run 10 epochs of mini-batch gradient descent: bug data space is 1 iteration ahead of parameter space.
LOSS_MBGD20, cost_MBGD20 = train_model_SGD(trainloader20, 10, w, b, lr, get_surface_MBGD20)

# Plot a graph that shows the LOSS results for all the methods.
plt.figure()
plt.plot(LOSS_GD, label="Batch Gradient Descent")
# plt.plot(LOSS_SGD, label="Stochastic Gradient Descent")
# plt.plot(LOSS_MBGD5, label="Mini-Batch Gradient Descent, Batch size=5")
# plt.plot(LOSS_MBGD10, label="Mini-Batch Gradient Descent, Batch size=10")
# plt.plot(LOSS_MBGD20, label="Mini-Batch Gradient Descent, Batch size=20")
plt.plot(cost_SGD, label="Stochastic Gradient Descent")
plt.plot(cost_MBGD5, label="Mini-Batch Gradient Descent, Batch size=5")
plt.plot(cost_MBGD10, label="Mini-Batch Gradient Descent, Batch size=10")
plt.plot(cost_MBGD20, label="Mini-Batch Gradient Descent, Batch size=20")
plt.legend()
plt.tight_layout()
plt.xlabel("Epoch/Iterations")
plt.ylabel("Cost")
plt.title('Loss value at each iteration')
plt.show()
