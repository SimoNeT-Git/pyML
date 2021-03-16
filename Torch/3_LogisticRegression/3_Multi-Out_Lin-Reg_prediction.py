#!/usr/bin/env python
# coding: utf-8

# In this lab, we will review how to make a prediction for Linear Regression with Multiple Output.

from torch import nn
import torch

# Set the random seed:
torch.manual_seed(1)


class linear_regression(nn.Module):
    def __init__(self, input_size, output_size):
        super(linear_regression, self).__init__()
        self.linear = nn.Linear(input_size, output_size)

    def forward(self, x):
        yhat = self.linear(x)
        return yhat


# create a linear regression object, as our input and output will be two we set the parameters accordingly
model = linear_regression(1, 10)
x = torch.tensor([1.0])
yhat = model(x)
print("\nFor an input x with shape", x.shape, "the shape of the prediction is", yhat.shape)

# we can see the parameters 
params = list(model.parameters())
print("\n\nThe randomly initialized weights of the model are", params[0].tolist(),
      "\nwith shape", params[0].shape[0], "x", params[0].shape[1])

# we can create a tensor with two rows representing one sample of data
x = torch.tensor([[1.0]])

# we can make a prediction
yhat = model(x)
print("\nFor an input x with shape", x.shape, "the shape of the prediction is", yhat.shape)

# each row in the following tensor represents a different sample
x = torch.tensor([[1.0], [1.0], [3.0]])

# we can make a prediction using multiple samples
yhat = model(x)
print("\nFor an input x with shape", x.shape, "the shape of the prediction is", yhat.shape)
