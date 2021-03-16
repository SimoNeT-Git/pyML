#!/usr/bin/env python
# coding: utf-8

# In this lab, you will train a model with PyTorch by using data that you created.
# The model only has 1 parameter: the slope.
#  - Make Some Data
#  - Create the Model and Cost Function (Total Loss)
#  - Train the Model

import torch
import matplotlib.pyplot as plt


# The class plot_diagram helps us to visualize the data space and the parameter space during training and has nothing
# to do with PyTorch.
class plot_diagram():

    # Constructor
    def __init__(self, X, Y, w, stop):
        start = w.data
        self.error = []
        self.parameter = []
        self.X = X.numpy()
        self.Y = Y.numpy()
        self.parameter_values = torch.arange(start, stop)
        self.Loss_function = [criterion(forward(X), Y) for w.data in self.parameter_values]
        w.data = start

    # Executor
    def __call__(self, Yhat, w, error, n):
        self.error.append(error)
        self.parameter.append(w.data)
        plt.figure(figsize=(7, 7))
        plt.subplot(212)
        plt.plot(self.X, self.Y, 'bo', label='Data Points')
        plt.plot(self.X, Yhat.detach().numpy(), 'r', label='Regression')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.legend()
        plt.ylim(-20, 20)
        plt.subplot(211)
        plt.title("Data Space (top) Estimated Line (bottom) Iteration " + str(n))
        plt.plot(self.parameter_values.numpy(), self.Loss_function, label='Loss Function')
        plt.plot(self.parameter, self.error, 'rx', label='Current Loss')
        plt.xlabel('Weight')
        plt.ylabel('Cost')
        plt.legend()
        plt.show()

    # # Destructor
    # def __del__(self):
    #     plt.close('all')


# #### Make Some Data

# Generate values from -3 to 3 that create a line f(x) with a slope of -3. This is the line you will estimate.
X = torch.arange(-3, 3, 0.1).view(-1, 1)
f = -3 * X

# Let us add some noise to the data f(x) in order to simulate the real data. Use torch.randn(X.size()) to generate
# Gaussian noise that is the same size as X and has a standard deviation of 0.1.
Y = f + 0.1 * torch.randn(X.size())

# Plot the data points
plt.figure()
plt.plot(X.numpy(), Y.numpy(), 'rx', label='Data Points')
plt.plot(X.numpy(), f.numpy(), label='Fit')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()


# #### Create the Model and Cost Function (Total Loss)

# In this section, let us create the model and the cost function (total loss) we are going to use to train the model
# and evaluate the result.

# First, define the forward function y = wx. (We will add the bias in the next lab.)
def forward(x):
    return w * x


# Define the cost or criterion function using MSE (Mean Square Error): 
def criterion(yhat, y):
    return torch.mean((yhat - y) ** 2)


# Define the learning rate lr
lr = 0.1

# Now, we create a model parameter by setting the argument requires_grad to True because the system must learn it.
w0 = -10.
w = torch.tensor(w0, requires_grad=True)


# #### Train the Model

# Let us define a function for training the model. The steps will be described in the comments.
def train_model(iter, weight, eta):
    # Create a plot_diagram object to visualize data space and parameter space for each iteration during training:
    gradient_plot = plot_diagram(X, Y, weight, stop=5)
    # Create empty list to save loss values at each iteration:
    loss_list = []
    for epoch in range(iter):
        # make the prediction
        Yhat = forward(X)
        # calculate the iteration
        loss = criterion(Yhat, Y)
        # store the loss into list
        loss_list.append(loss.item())

        # plot the diagram for us to have a better idea
        gradient_plot(Yhat, weight, loss.item(), epoch)

        # backward pass: compute gradient of the loss with respect to all the learnable parameters
        loss.backward()
        # updata parameters
        weight.data = weight.data - eta * weight.grad.data
        # zero the gradients before running the backward pass
        weight.grad.data.zero_()

    return loss_list


# Let us try to run 4 iterations of gradient descent and save the loss for each iteration in a list LOSS.
LOSS = train_model(4, w, lr)

# Now lets try to initialize w with a value of -15.0.
w0_2 = -15.
w = torch.tensor(w0_2, requires_grad=True)

# Lets run 4 iterations of gradient descent:
LOSS2 = train_model(4, w, lr)

# Plot an overlay of the list LOSS2 and LOSS.
plt.figure(figsize=(6, 5))
plt.plot(LOSS, label="initial w is {}".format(int(w0)))
plt.plot(LOSS2, label="initial w is {}".format(int(w0_2)))
plt.legend()
plt.tight_layout()
plt.xlabel("Epoch/Iterations")
plt.ylabel("Cost")
plt.title('Loss value at each iteration')
plt.show()
# What does this tell you about the parameter value?
