#!/usr/bin/env python
# coding: utf-8

# In this lab, you will train a model with PyTorch by using the data that we created. The model will have the slope and
# bias. And we will review how to make a prediction in several different ways by using PyTorch.
#  - Make Some Data
#  - Create the Model and Cost Function (Total Loss)
#  - Train the Model

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

# Start with generating values from -3 to 3 that create a line with a slope of 1 and a bias of -1. This is the line that
# you need to estimate.

# Create f(X) with a slope of 1 and a bias of -1
X = torch.arange(-3, 3, 0.1).view(-1, 1)
f = 1 * X - 1

# Now, add some noise to the data:
Y = f + 0.1 * torch.randn(X.size())

# Plot out the line and the points with noise
plt.figure()
plt.plot(X.numpy(), Y.numpy(), 'rx', label='Data Points')
plt.plot(X.numpy(), f.numpy(), label='Fit')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()


# #### Create the Model and Cost Function (Total Loss)

# Define the forward function:
def forward(x):
    return w * x + b


# Define the cost or criterion function (MSE):
def criterion(yhat, y):
    return torch.mean((yhat - y) ** 2)


# Create a plot_error_surfaces object to visualize the data space and the parameter space during training:
get_surface = plot_error_surfaces(15, 15, X, Y, 30)


# #### Train the Model

# Create model parameters w, b (for y = wx + b)  by setting the argument requires_grad to True because we must learn it
# using the data.
w0 = -15.0
b0 = -10.0
w = torch.tensor(w0, requires_grad=True)
b = torch.tensor(b0, requires_grad=True)

# Set the learning rate to 0.1 and create an empty list LOSS for storing the loss for each iteration.
lr = 0.1


# Define train_model function for train the model.
def train_model(n_iter, weight, bias, eta, surf):
    # Create empty list to save loss values at each iteration:
    loss_list = []
    for epoch in range(n_iter):
        # make a prediction
        Yhat = forward(X)
        # calculate the loss 
        loss = criterion(Yhat, Y)
        # store the loss
        loss_list.append(loss)

        # Section for plotting
        surf.set_para_loss(weight.data.tolist(), bias.data.tolist(), loss.tolist())
        if epoch % 3 == 0:
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


# Run 15 iterations of gradient descent: bug data space is 1 iteration ahead of parameter space
LOSS = train_model(15, w, b, lr, get_surface)

# Plot total loss/cost surface with loss values for different parameters in red:
get_surface.final_plot()

# Now lets try to set the learning rate to 0.2.
w = torch.tensor(w0, requires_grad=True)
b = torch.tensor(b0, requires_grad=True)
lr2 = 0.2

# Create a plot_error_surfaces object to visualize the data space and the parameter space during training:
get_surface2 = plot_error_surfaces(15, 15, X, Y, 30, go=False)

# Run 15 iterations of gradient descent
LOSS2 = train_model(15, w, b, lr2, get_surface2)

# Plot the LOSS and LOSS2 in order to compare the Total Loss
plt.figure()
plt.plot(LOSS, label="learning rate is {}".format(lr))
plt.plot(LOSS2, label="learning rate is {}".format(lr2))
plt.legend()
plt.tight_layout()
plt.xlabel("Epoch/Iterations")
plt.ylabel("Cost")
plt.title('Loss value at each iteration')
plt.show()
