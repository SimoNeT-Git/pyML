#!/usr/bin/env python
# coding: utf-8

# In this lab, we will cover logistic regression using PyTorch.
#   - Logistic Function
#   - Build a Logistic Regression Using nn.Sequential
#    - Build Custom Modules

import torch.nn as nn
import torch
import matplotlib.pyplot as plt 

# Set the random seed:
torch.manual_seed(2)


# #### Logistic Function

# Create a tensor ranging from -100 to 100:
z = torch.arange(-100, 100, 0.1).view(-1, 1)

# Create a sigmoid object:
sig = nn.Sigmoid()

# Apply the element-wise function Sigmoid with the object:
yhat = sig(z)

# Plot the results: 
plt.figure()
plt.plot(z.numpy(), yhat.numpy())
plt.xlabel('z')
plt.ylabel('yhat')
plt.title('Logistic Function: Sigmoid with nn.Sigmoid()')
plt.show()

# Apply the element-wise Sigmoid from the function module and plot the results:
yhat = torch.sigmoid(z)
plt.figure()
plt.plot(z.numpy(), yhat.numpy())
plt.xlabel('z')
plt.ylabel('yhat')
plt.title('Logistic Function: Sigmoid with torch.sigmoid()')
plt.show()


# #### Build a Logistic Regression with nn.Sequential

# Create a 1x1 tensor where x represents one data sample with one dimension, and 2x1 tensor X represents two data
# samples of one dimension:
x = torch.tensor([[1.0]])
X = torch.tensor([[1.0], [100]])

# Create a logistic regression object with the nn.Sequential model with a one-dimensional input:
model = nn.Sequential(nn.Linear(1, 1), nn.Sigmoid())

# In this case, the parameters are randomly initialized. You can view them the following ways:
params = list(model.parameters())
params_dict = model.state_dict()
print("\nCase 1:")
print("The weight is", params[0].tolist(), "while the bias is", params[1].tolist())
print("We can also print them with the state_dict(): the weight is", params_dict['0.weight'].tolist(),
      "while the bias is", params_dict['0.bias'].tolist())

# Make a prediction with one sample:
yhat = model(x)
print("\nFor an input x=", x.tolist(), "with shape", list(x.shape), "the shape of the prediction yhat=", yhat.tolist(),
      "is", list(yhat.shape))

# Make a prediction with multiple samples:
Yhat = model(X)
print("\nFor an input x=", X.tolist(), "with shape", list(X.shape), "the shape of the prediction yhat=", Yhat.tolist(),
      "is", list(Yhat.shape))


# Create a 1x2 tensor where x represents one data sample with one dimension, and 2x3 tensor X represents one data sample
# of two dimensions:
x = torch.tensor([[1.0, 1.0]])
X = torch.tensor([[1.0, 1.0], [1.0, 2.0], [1.0, 3.0]])

# Create a logistic regression object with the nn.Sequential model with a two-dimensional input:
model = nn.Sequential(nn.Linear(2, 1), nn.Sigmoid())

# In this case, the parameters are randomly initialized. You can view them the following ways:
params = list(model.parameters())
params_dict = model.state_dict()
print("\nCase 2:")
print("The weight is", params[0].tolist(), "while the bias is", params[1].tolist())
print("We can also print them with the state_dict(): the weight is", params_dict['0.weight'].tolist(),
      "while the bias is", params_dict['0.bias'].tolist())

# Make a prediction with one sample:
yhat = model(x)
print("\nFor an input x=", x.tolist(), "with shape", list(x.shape), "the shape of the prediction yhat=", yhat.tolist(),
      "is", list(yhat.shape))

# Make a prediction with multiple samples:
Yhat = model(X)
print("\nFor an input x=", X.tolist(), "with shape", list(X.shape), "the shape of the prediction yhat=", Yhat.tolist(),
      "is", list(Yhat.shape))


# #### Build Custom Modules

# In this section, you will build a custom Module or class for logistic regression. The model or object function is
# identical to using nn.Sequential
class logistic_regression(nn.Module):
    
    # Constructor
    def __init__(self, n_inputs):
        super(logistic_regression, self).__init__()
        self.linear = nn.Linear(n_inputs, 1)
    
    # Prediction
    def forward(self, x):
        yhat = torch.sigmoid(self.linear(x))
        return yhat


# Create a model to predict one dimension:
model = logistic_regression(1)
params = list(model.parameters())
params_dict = model.state_dict()
print("\nCase 3:")
print("The weight is", params[0].tolist(), "while the bias is", params[1].tolist())
print("We can also print them with the state_dict(): the weight is", params_dict['linear.weight'].tolist(),
      "while the bias is", params_dict['linear.bias'].tolist())

# Create a 1x1 tensor where x represents one data sample with one dimension, and 3x1 tensor where $X$ represents one
# data sample of one dimension:
x = torch.tensor([[1.0]])
X = torch.tensor([[-100], [0], [100.0]])

# Make a prediction with one sample:
yhat = model(x)
print("\nFor an input x=", x.tolist(), "with shape", list(x.shape), "the shape of the prediction yhat=", yhat.tolist(),
      "is", list(yhat.shape))

# Make a prediction with multiple samples:
Yhat = model(X)
print("\nFor an input x=", X.tolist(), "with shape", list(X.shape), "the shape of the prediction yhat=", Yhat.tolist(),
      "is", list(Yhat.shape))

# Create a logistic regression object with a function with two inputs:
model = logistic_regression(2)
params = list(model.parameters())
params_dict = model.state_dict()
print("\nCase 4:")
print("The weight is", params[0].tolist(), "while the bias is", params[1].tolist())
print("We can also print them with the state_dict(): the weight is", params_dict['linear.weight'].tolist(),
      "while the bias is", params_dict['linear.bias'].tolist())

# Create a 1x2 tensor where x represents one data sample with one dimension, and 3x2 tensor X represents one data
# sample of one dimension:
x = torch.tensor([[1.0, 2.0]])
X = torch.tensor([[100, -100], [0.0, 0.0], [-100, 100]])

# Make a prediction with one sample:
yhat = model(x)
print("\nFor an input x=", x.tolist(), "with shape", list(x.shape), "the shape of the prediction yhat=", yhat.tolist(),
      "is", list(yhat.shape))

# Make a prediction with multiple samples:
Yhat = model(X)
print("\nFor an input x=", X.tolist(), "with shape", list(X.shape), "the shape of the prediction yhat=", Yhat.tolist(),
      "is", list(Yhat.shape))
