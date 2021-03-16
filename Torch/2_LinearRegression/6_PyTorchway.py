#!/usr/bin/env python
# coding: utf-8

# In this lab, you will create a model the PyTroch way, this will help you as models get more complicated.
#  - Make Some Data
#  - Create the Model and Cost Function the PyTorch way
#  - Train the Model: Batch Gradient Descent

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
    def set_para_loss(self, model, loss):
        self.n = self.n + 1
        self.W.append(list(model.parameters())[0].item())
        self.B.append(list(model.parameters())[1].item())
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
from torch.utils.data import Dataset, DataLoader

# Set random seed:
torch.manual_seed(1)


# Generate values from -3 to 3 that create a line with a slope of 1 and a bias of -1. This is the line that you need to
# estimate. Add some noise to the data:
class Data(Dataset):

    # Constructor
    def __init__(self):
        self.x = torch.arange(-3, 3, 0.1).view(-1, 1)
        self.f = 1 * self.x - 1
        self.y = self.f + 0.1 * torch.randn(self.x.size())
        self.len = self.x.shape[0]

    # Getter
    def __getitem__(self, index):
        return self.x[index], self.y[index]

    # Get Length
    def __len__(self):
        return self.len


# Create a dataset object:
dataset = Data()

# Plot out the data and the line.
plt.figure()
plt.plot(dataset.x.numpy(), dataset.y.numpy(), 'rx', label='Data Points')
plt.plot(dataset.x.numpy(), dataset.f.numpy(), label='Fit')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()

# #### Create the Model and Total Loss Function (Cost)
from torch import nn, optim


# Create a linear regression class
class linear_regression(nn.Module):

    # Constructor
    def __init__(self, input_size, output_size):
        super(linear_regression, self).__init__()
        self.linear = nn.Linear(input_size, output_size)

    # Prediction
    def forward(self, x):
        yhat = self.linear(x)
        return yhat


# We will use PyTorch build-in functions to create a criterion function; this calculates the total loss or cost
criterion = nn.MSELoss()

# Create a linear regression object and optimizer object, the optimizer object will use the linear regression object.
lr = 0.01
model = linear_regression(1, 1)
optimizer = optim.SGD(model.parameters(), lr=lr)
params = list(model.parameters())  # or, equally, optimizer.param_groups[0]['params']
print("\nFirst model parameters at original initialization:")
print("Weight =", params[0][0])
print("Bias =", params[1][0])
print("Learning Rate =", optimizer.param_groups[0]['lr'])
# Remember: to construct an optimizer you have to give it an iterable containing the parameters i.e. provide
# model.parameters() as an input to the object constructor

# Similar to the model, the optimizer has a state dictionary:
print("\nOptimizer parameters can also be accessed: ", optimizer.state_dict())
# Many of the keys correspond to more advanced optimizers.

# Create a Dataloader object:
trainloader = DataLoader(dataset=dataset)  # by default, batch_size=1

# PyTorch randomly initialises your model parameters. If we use those parameters, the result will not be very insightful
# as convergence will be extremely fast. So we will initialise the parameters such that they will take longer to
# converge, i.e. look cool

# Customize the weight and bias
w0 = -15.
b0 = -10.
model.state_dict()['linear.weight'][0] = w0
model.state_dict()['linear.bias'][0] = b0
params = list(model.parameters())  # or, equally, optimizer.param_groups[0]['params']
print("\nFirst model parameters at new initialization:")
print("Weight =", params[0].tolist())
print("Bias =", params[1].tolist())
print("Learning Rate =", optimizer.param_groups[0]['lr'])

# Create a plotting object, not part of PyTroch, just used to help visualize
get_surface = plot_error_surfaces(15, 13, dataset.x, dataset.y, 30)


# #### Train the Model via Batch Gradient Descent

# Train Model
def train_model_BGD(XY, n_iter, reg_model, optim, surf):
    Xvar, Yvar = zip(*XY)
    Xvar, Yvar = (torch.stack(Xvar).view(-1, 1), torch.stack(Yvar).view(-1, 1))

    loss_list = []
    cost_list = []

    for epoch in range(n_iter):
        Yhat = reg_model(Xvar)
        loss_list.append(criterion(Yhat, Yvar).tolist())
        tot_loss = 0
        for x, y in XY:
            # make a prediction
            yhat = reg_model(x)
            # calculate the loss
            loss = criterion(yhat, y)
            # update total loss summing the loss value for each data point
            tot_loss += loss.item()

            # Section for plotting
            surf.set_para_loss(reg_model, loss.tolist())

            # zero the gradients before running the backward pass
            optim.zero_grad()
            # compute gradient of the loss with respect to all the learnable parameters
            loss.backward()
            # update parameters slope (weight) and bias through optimization step
            optim.step()

        # store the mean (over the dataset) of the total loss (computed at each epoch on the whole dataset)
        cost_list.append(tot_loss/len(XY))

        # plot surface and data space after each epoch
        surf.plot_ps()

    return loss_list


# Run 10 epochs of stochastic gradient descent: bug data space is 1 iteration ahead of parameter space.
LOSS_GD, cost_GD = train_model_BGD(trainloader, 10, model, optimizer, get_surface)
print("\nModel parameters after training:")
print("Weight =", model.state_dict()['linear.weight'][0])
print("Bias =", model.state_dict()['linear.bias'][0])


# Let's clarify the process. The model takes x to produce an estimate yhat, it will then be compared to the actual y
# with the loss function.
# When we call backward() on the loss function, it will handle the differentiation. Calling the method step on the
# optimizer object it will update the parameters as they were inputs when we constructed the optimizer object.


# Now lets train the model with lr = 0.1. Use optimizer and the following given variables.
lr2 = 0.1
model2 = linear_regression(1, 1)
model2.state_dict()['linear.weight'][0] = w0
model2.state_dict()['linear.bias'][0] = b0
optimizer2 = optim.SGD(model2.parameters(), lr=lr2)
print("\nSecond model parameters at initialization:")
print("Weight =", w0)
print("Bias =", b0)
print("Learning Rate =", optimizer2.param_groups[0]['lr'])
get_surface2 = plot_error_surfaces(15, 13, dataset.x, dataset.y, 30, go=False)
LOSS_GD2, cost_GD2 = train_model_BGD(trainloader, 10, model2, optimizer2, get_surface2)
print("\nModel parameters after training:")
print("Weight =", model2.state_dict()['linear.weight'][0])
print("Bias =", model2.state_dict()['linear.bias'][0])

# Plot the loss for each epoch:
plt.figure()
# plt.plot(LOSS_GD, label="GD with learning_rate={}".format(lr))
# plt.plot(LOSS_GD2, label="GD with learning_rate={}".format(lr2))
plt.plot(cost_GD, label="GD with learning_rate={}".format(lr))
plt.plot(cost_GD2, label="GD with learning_rate={}".format(lr2))
plt.legend()
plt.tight_layout()
plt.xlabel("Epoch/Iterations")
plt.ylabel("Cost")
plt.title('Loss value at each iteration')
plt.show()
