#!/usr/bin/env python
# coding: utf-8

# In this lab, you will deal with several problems associated with optimization and see how momentum can improve your
# results.
#   - Saddle Points
#   - Local Minima
#   - Noise

import torch
import torch.nn as nn
import matplotlib.pylab as plt
torch.manual_seed(0)


# This function will plot a cubic function and the parameter values obtained via Gradient Descent.
def plot_cubic(w, optimizer, title='Cubic Loss Function'):
    LOSS = []
    # parameter values 
    W = torch.arange(-4, 4, 0.1)
    # plot the loss function
    for w.state_dict()['linear.weight'][0] in W:
        LOSS.append(cubic(w(torch.tensor([[1.0]]))).item())
    w.state_dict()['linear.weight'][0] = 4.0
    n_epochs = 10
    parameter = []
    loss_list = []

    # Epochs. Use PyTorch custom module to implement a polynomial function.
    for n in range(n_epochs):
        optimizer.zero_grad()
        loss = cubic(w(torch.tensor([[1.0]])))
        loss_list.append(loss)
        parameter.append(w.state_dict()['linear.weight'][0].detach().data.item())
        loss.backward()
        optimizer.step()

    # Plotting
    plt.figure()
    plt.plot(W.numpy(), LOSS, label='objective function')
    plt.plot(parameter, loss_list, 'ro', label='parameter values')
    plt.xlabel('w')
    plt.ylabel('L(w)')
    plt.title(title)
    plt.legend()
    plt.show()


# This function will plot a 4th order function and the parameter values obtained via Gradient Descent. You can also add
# Gaussian noise with a standard deviation determined by the parameter std.
def plot_fourth_order(w, optimizer, std=0, title='Fourth-Order Loss Function'):
    W = torch.arange(-4, 6, 0.1)
    LOSS = []
    for w.state_dict()['linear.weight'][0] in W:
        LOSS.append(fourth_order(w(torch.tensor([[1.0]]))).item())
    w.state_dict()['linear.weight'][0] = 6
    n_epochs = 100
    parameter = []
    loss_list = []

    # Epochs
    for n in range(n_epochs):
        optimizer.zero_grad()
        loss = fourth_order(w(torch.tensor([[1.0]]))) + std * torch.randn(1, 1)
        loss_list.append(loss)
        parameter.append(w.state_dict()['linear.weight'][0].detach().data.item())
        loss.backward()
        optimizer.step()

    # Plotting
    plt.figure()
    plt.plot(W.numpy(), LOSS, label='objective function')
    plt.plot(parameter, loss_list, 'ro', label='parameter values')
    plt.xlabel('w')
    plt.ylabel('L(w)')
    plt.title(title)
    plt.legend()
    plt.show()


# This is a custom module. It will behave like a linear model with a single parameter value (only weights, no bias).
# We do it this way so we can use PyTorch's build-in optimizers.
class one_param(nn.Module):

    # Constructor
    def __init__(self, input_size, output_size):
        super(one_param, self).__init__()
        self.linear = nn.Linear(input_size, output_size, bias=False)

    # Prediction
    def forward(self, x):
        yhat = self.linear(x)
        return yhat


# We create an object w, when we call the object with an input of one, it will behave like an individual parameter
# value. i.e w(1) is analogous to w.
w = one_param(1, 1)


# #### Saddle Points

# Let's create a cubic function with Saddle points
def cubic(yhat):
    return yhat ** 3


# We create an optimizer with no momentum term
optimizer = torch.optim.SGD(w.parameters(), lr=0.01, momentum=0)
# We run several iterations of stochastic gradient descent and plot the results.
print("\nWe see that for cubic loss function and optimizer with learning rate=0.01 and NO momentum\n"
      "the parameter values get stuck in the saddle point.")
plot_cubic(w, optimizer, title='Cubic loss for optimizer with\nlearning rate=0.01 and no momentum')

# We create an optimizer with momentum term of 0.9
optimizer = torch.optim.SGD(w.parameters(), lr=0.01, momentum=0.9)
# We run several iterations of stochastic gradient descent with momentum and plot the results.
print("\nWe see that for cubic loss function and optimizer with learning rate=0.01 and momentum=0.9\n"
      "the parameter values do not get stuck in the saddle point.")
plot_cubic(w, optimizer, title='Cubic loss for optimizer with\nlearning rate=0.01 and momentum=0.9')


# #### Local Minima

# In this section, we will create a fourth order polynomial with a local minimum at 4 and a global minimum a -2. We will
# then see how the momentum parameter affects convergence to a global minimum. The fourth order polynomial is given by:
def fourth_order(yhat):
    out = torch.mean(2 * (yhat ** 4) - 9 * (yhat ** 3) - 21 * (yhat ** 2) + 88 * yhat + 48)
    return out


# We create an optimizer with no momentum term. We run several iterations of stochastic gradient descent and plot the
# results. We see the parameter values get stuck in the local minimum.
optimizer = torch.optim.SGD(w.parameters(), lr=0.001)
# We run several iterations of stochastic gradient descent and plot the results.
print("\nWe see that for fourth-order loss function and optimizer with learning rate=0.001 and NO momentum\n"
      "the parameter values get stuck in the local minima of the loss function.")
plot_fourth_order(w, optimizer, title='Fourth-order loss for optimizer with\nlearning rate=0.001 and no momentum')

# We create an optimizer with a momentum term of 0.9. We run several iterations of stochastic gradient descent and plot
# the results. We see the parameter values reach a global minimum.
optimizer = torch.optim.SGD(w.parameters(), lr=0.001, momentum=0.9)
# We run several iterations of stochastic gradient descent with momentum and plot the results.
print("\nWe see that for fourth-order loss function and optimizer with learning rate=0.001 and momentum=0.9\n"
      "the parameter values do not get stuck in the local minima of the loss function\n"
      "and converge to the global minima.")
plot_fourth_order(w, optimizer, title='Fourth-order loss for optimizer with\nlearning rate=0.001 and momentum=0.9')


# #### Noise

# In this section, we will create a fourth order polynomial with a local minimum at 4 and a global minimum a -2, but we
# will add noise to the function when the Gradient is calculated. We will then see how the momentum parameter affects
# convergence to a global minimum.

# Make the prediction without momentum when there is noise
optimizer = torch.optim.SGD(w.parameters(), lr=0.001)
# We run several iterations of stochastic gradient descent and plot the results.
print("\nWe see that for fourth-order loss function and optimizer with learning rate=0.001 and NO momentum\n"
      "the parameter values, affected by noise with std=10, get stuck in the local minima of the loss function.")
plot_fourth_order(w, optimizer, std=10,
                  title='Fourth-order loss for noisy weights (std=10) and\n'
                        'optimizer with learning rate=0.001 and no momentum')

# Make the prediction with momentum when there is noise
optimizer = torch.optim.SGD(w.parameters(), lr=0.001, momentum=0.9)
# We run several iterations of stochastic gradient descent with momentum and plot the results.
print("\nWe see that for fourth-order loss function and optimizer with learning rate=0.001 and momentum=0.9\n"
      "the parameter values, affected by noise with std=10, do not get stuck in the local minima of the loss function\n"
      "and converge to the global minima.")
plot_fourth_order(w, optimizer, std=10,
                  title='Fourth-order loss for noisy weights (std=10) and\n'
                        'optimizer with learning rate=0.001 and momentum=0.9')
