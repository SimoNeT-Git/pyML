#!/usr/bin/env python
# coding: utf-8

# In this lab, you will review how to make a prediction in several different ways by using PyTorch.
#  - Prediction
#  - Class Linear
#  - Build Custom Modules

from torch import nn
import torch

torch.manual_seed(1)

# #### Prediction

# Set the weight and bias. Note that torch.mm uses matrix multiplication instead of scalar multiplication.
w0 = [[2.0], [3.0]]
b0 = [[1.0]]
w = torch.tensor(w0, requires_grad=True)
b = torch.tensor(b0, requires_grad=True)
print("The weights of the custom-made linear regression model are", w.tolist(),
      "\nwith shape", w.shape[0], "x", w.shape[1], "while the bias is", b.tolist())


# Define Prediction Function. The torch.mm function implements a matrix multiplication.
def forward(x):
    yhat = torch.mm(x, w) + b
    return yhat


# If we input a 1x2 tensor x, because we have a 2x1 tensor as w, we will get a 1x1 tensor yhat:
x = torch.tensor([[1.0, 2.0]])
yhat = forward(x)
print("\nThe linear prediction to x=", x.tolist(), "with shape", x.shape[0], "x", x.shape[1],
      "\nis y_hat=", yhat.tolist(), "with shape", yhat.shape[0], "x", yhat.shape[1])

# Each row of the following tensor represents a sample:
X = torch.tensor([[1.0, 1.0], [1.0, 2.0], [1.0, 3.0]])

# Make the prediction of X
yhat = forward(X)
print("\nThe linear prediction to x=", X.tolist(), "with shape", X.shape[0], "x", X.shape[1],
      "\nis y_hat=", yhat.tolist(), "with shape", yhat.shape[0], "x", yhat.shape[1])


# #### Class Linear

# We can use the linear class to make a prediction. You'll also use the linear class to build more complex models.
# Make a linear regression model using built-in function
model = nn.Linear(in_features=2, out_features=1)
params = list(model.parameters())
print("\n\nThe weights of the torch.nn.linear model randomly initialized are", params[0].tolist(),
      "\nwith shape", params[0].shape[0], "x", params[0].shape[1], "while the bias is", params[1].tolist())

# Make a prediction with the first sample:
yhat = model(x)
print("\nThe linear prediction to x=", x.tolist(), "with shape", x.shape[0], "x", x.shape[1],
      "\nis y_hat=", yhat.tolist(), "with shape", yhat.shape[0], "x", yhat.shape[1])

# Predict with multiple samples X:
yhat = model(X)
print("\nThe linear prediction to x=", X.tolist(), "with shape", X.shape[0], "x", X.shape[1],
      "\nis y_hat=", yhat.tolist(), "with shape", yhat.shape[0], "x", yhat.shape[1])


# #### Build Custom Modules

# Now, you'll build a custom module. You can make more complex models by using this method later.
# Create linear_regression Class
class linear_regression(nn.Module):

    # Constructor
    def __init__(self, input_size, output_size):
        super(linear_regression, self).__init__()
        self.linear = nn.Linear(input_size, output_size)

    # Prediction function
    def forward(self, x):
        yhat = self.linear(x)
        return yhat


# Build a linear regression object. The input feature size is 2.
model = linear_regression(2, 1)
params = list(model.parameters())
print("\n\nThe weights of the custom linear_regression class that uses the torch.nn.linear model "
      "randomly initialized are", params[0].tolist(), "\nwith shape", params[0].shape[0], "x", params[0].shape[1],
      "while the bias is", params[1].tolist())

# You can also see the parameters by using the state_dict() method:
print("Another way to see the parameters is with model.state_dict(): ", model.state_dict())

# Now we input a 1x2 tensor, and we will get a 1x1 tensor.
yhat = model(x)
print("\nThe linear prediction to x=", x.tolist(), "with shape", x.shape[0], "x", x.shape[1],
      "\nis y_hat=", yhat.tolist(), "with shape", yhat.shape[0], "x", yhat.shape[1])

# Make a prediction for multiple samples:
yhat = model(X)
print("\nThe linear prediction to x=", X.tolist(), "with shape", X.shape[0], "x", X.shape[1],
      "\nis y_hat=", yhat.tolist(), "with shape", yhat.shape[0], "x", yhat.shape[1])

# Build a model or object of type linear_regression to predict the tensor X:
X = torch.tensor([[11.0, 12.0, 13, 14], [11, 12, 13, 14]])
model = linear_regression(4, 1)
yhat = model(X)
print("\nThe linear prediction to x=", X.tolist(), "with shape", X.shape[0], "x", X.shape[1],
      "\nis y_hat=", yhat.tolist(), "with shape", yhat.shape[0], "x", yhat.shape[1])
