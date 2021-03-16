#!/usr/bin/env python
# coding: utf-8

# In this lab, we will  review how to make a prediction in several different ways by using PyTorch.
#  - Prediction
#  - Class Linear
#  - Build Custom Modules

import torch

# #### Simple Prediction
print("\n\nSimple predictions:")

# Let us create the following expressions: y = wx + b
# First, define the parameters: w = 2 and b = -1 --> y = 2x - 1
w = torch.tensor(2.0, requires_grad=True)
b = torch.tensor(-1.0, requires_grad=True)


# Then, define the function forward(x, w, b) that makes the prediction:
# Function forward(x) for prediction
def forward(x):
    yhat = w * x + b
    return yhat


# Let's make the following prediction at x = 1: y = - 1 + 2(1) = 1
x = torch.tensor([[1.0]])
yhat = forward(x)
print("The prediction of y = 2x - 1 at x = 1 is: ", yhat)

# Now, let us try to make the prediction for multiple inputs:
# First, we construct the x tensor.
x = torch.tensor([[1.0], [2.0], [3.0]])

# Now make the prediction at x = [[1.0], [2.0], [3.0]]: y1 = 1 and y2 = 3, y3 = 5
yhat = forward(x)
print("The prediction at x = [1, 2, 3] is: ", yhat)


# #### Class Linear
print("\n\nLinear class:")

# The linear class can be used to make a prediction. We can also use the linear class to build more complex models.
from torch.nn import Linear

# Set the random seed because the parameters are randomly initialized:
torch.manual_seed(1)

# Let us create the linear object by using the constructor. The parameters are randomly created. Let us print them out
# to see w (weight) and b (bias) values. The parameters of an torch.nn.Module model are contained in the model’s
# parameters accessed with lr.parameters():
lr = Linear(in_features=1, out_features=1, bias=True)
print("\nParameters w and b are: ", list(lr.parameters()))

# A method state_dict() Returns a Python dictionary object corresponding to the layers of each parameter tensor.
print("Python dictionary: ", lr.state_dict())
print("keys: ", lr.state_dict().keys())
print("values: ", lr.state_dict().values())
# The keys correspond to the name of the attributes and the values correspond to the parameter value.

# Another way to check parameter values is:
print("\nWeight value w is:", float(lr.weight))
print("Bias value b is:", float(lr.bias))

# Now let us make a single prediction at x = [[1.0]] --> y = 0.5153*1 - 0.4414 = 0.0739
x = torch.tensor([[1.0]])
yhat = lr(x)
print("\nThe prediction at x = 1 is: ", yhat)

# Similarly, you can make multiple predictions:
x = torch.tensor([[1.0], [2.0], [3.0]])
yhat = lr(x)
print("The prediction at x = [1, 2, 3] is: ", yhat)


# #### Build Custom Modules
print("\n\nCustom modules:")

from torch import nn

# Now, let's build a custom module. We can make more complex models by using this method later on.


# Let us define the class Customize Linear Regression
class LR(nn.Module):

    # Constructor
    def __init__(self, input_size, output_size):
        # Inherit from parent
        super(LR, self).__init__()
        self.linear = nn.Linear(input_size, output_size)

    # Prediction function
    def forward(self, x):
        out = self.linear(x)
        return out


# Create an object by using the constructor. Print out the parameters we get and the model.
lr = LR(1, 1)
print("The parameters: ", list(lr.parameters()))
print("Linear model: ", lr.linear)

# the parameters are also stored in an ordered dictionary :
print("Python dictionary: ", lr.state_dict())
print("keys: ", lr.state_dict().keys())
print("values: ", lr.state_dict().values())

# Let us try to make a prediction of a single input sample.
x = torch.tensor([[1.0]])
yhat = lr(x)
print("\nThe prediction at x = 1 is: ", yhat)

# Now, let us try another example with multiple samples.
x = torch.tensor([[1.0], [2.0], [3.0]])
yhat = lr(x)
print("The prediction at x = [1, 2, 3] is: ", yhat)
