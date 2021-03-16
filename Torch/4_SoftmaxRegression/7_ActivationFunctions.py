#!/usr/bin/env python
# coding: utf-8

# In this lab, you will cover logistic regression and activation functions by using PyTorch.
#   - Logistic Function
#   - Tanh
#   - Relu
#   - Compare Activation Functions

import torch.nn as nn
import torch
import matplotlib.pyplot as plt

torch.manual_seed(2)


# #### Logistic Function

# Create a tensor ranging from -10 to 10:
z = torch.arange(-10, 10, 0.1).view(-1, 1)

# Create a sigmoid object and use it to make the prediction on z
SIG = nn.Sigmoid()
yhat = SIG(z)

# Plot the result:
plt.figure()
plt.plot(z.numpy(), yhat.numpy())
plt.xlabel('x')
plt.ylabel('yhat')
plt.title('Sigmoid Function')
plt.show()

# Note: For custom modules, call the sigmoid object from the torch (nn.functional for the old version) for making
# prediction, which applies the element-wise sigmoid from the function module:
yhat = torch.sigmoid(z)  # nn.functional.sigmoid(z)


# #### Tanh

# Create a tanh object and use it to make the prediction on z
TANH = nn.Tanh()
yhat = TANH(z)

# Plot the result:
plt.figure()
plt.plot(z.numpy(), yhat.numpy())
plt.xlabel('x')
plt.ylabel('yhat')
plt.title('Tanh Function')
plt.show()

# Note: For custom modules, call the Tanh object from the torch (nn.functional for the old version) for making
# prediction, which applies the element-wise sigmoid from the function module:
yhat = torch.tanh(z)  # or nn.functional.tanh(z)


# #### Relu

# Create a relu object and use it to make the prediction on z
RELU = nn.ReLU()
yhat = RELU(z)

# Plot the result
plt.figure()
plt.plot(z.numpy(), yhat.numpy())
plt.xlabel('x')
plt.ylabel('yhat')
plt.title('ReLu Function')
plt.show()

# Note: For custom modules, call the relu object from the torch (nn.functional for the old version) for making
# prediction, which applies the element-wise sigmoid from the function module:
yhat = torch.relu(z)  # nn.functional.relu(z)


# #### Compare Activation Functions

# Plot the results to compare the activation functions
x = torch.arange(-2, 2, 0.1).view(-1, 1)
plt.figure()
plt.plot(x.numpy(), torch.sigmoid(x).numpy(), label='sigmoid')
plt.plot(x.numpy(), torch.tanh(x).numpy(), label='tanh')
plt.plot(x.numpy(), torch.relu(x).numpy(), label='relu')
plt.xlabel('x')
plt.ylabel('yhat')
plt.title('Comparison of Activation Functions\nfor x ranging from -2 to 2')
plt.legend()
plt.show()

# Compare the activation functions with a tensor in the range (-1, 1)
x = torch.arange(-1, 1, 0.1).view(-1, 1)
plt.figure()
plt.plot(x.numpy(), torch.sigmoid(x).numpy(), label='sigmoid')
plt.plot(x.numpy(), torch.tanh(x).numpy(), label='tanh')
plt.plot(x.numpy(), torch.relu(x).numpy(), label='relu')
plt.xlabel('x')
plt.ylabel('yhat')
plt.title('Comparison of Activation Functions\nfor x ranging from -1 to 1')
plt.legend()
plt.show()
