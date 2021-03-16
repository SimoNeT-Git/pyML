#!/usr/bin/env python
# coding: utf-8

# In this lab, you will learn to select the best learning rate by using validation data.
#  - Make Some Data
#  - Create a Linear Regression Object, Data Loader and Criterion Function
#  - Different learning rates and Data Structures to Store results for Different Hyperparameters
#  - Train different modules for different Hyperparameters
#  - View Results

import torch
import numpy as np
import matplotlib.pyplot as plt
from torch import nn, optim

# #### Make Some Data
from torch.utils.data import Dataset, DataLoader


# First, we'll create some artificial data in a dataset class. The class will include the option to produce training
# data or validation data. The training data will include outliers.
class Data(Dataset):

    # Constructor
    def __init__(self, train=True):
        self.x = torch.arange(-3, 3, 0.1).view(-1, 1)
        self.f = -3 * self.x + 1
        self.y = self.f + 0.1 * torch.randn(self.x.size())
        self.len = self.x.shape[0]

        # outliers
        if train:
            self.y[0] = 0
            self.y[50:55] = 20
        else:
            pass

    # Getter
    def __getitem__(self, index):
        return self.x[index], self.y[index]

    # Get Length
    def __len__(self):
        return self.len


# Create two objects: one that contains training data and a second that contains validation data. Assume that the
# training data has the outliers.
train_data = Data()
val_data = Data(train=False)

# Overlay the training points in red over the function that generated the data. Notice the outliers at x=-3 and around
# x=2:
plt.figure()
plt.plot(train_data.x.numpy(), train_data.y.numpy(), 'xr', label="Data Points")
plt.plot(train_data.x.numpy(), train_data.f.numpy(), label="True Function")
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()


# #### Create a Linear Regression Object,  Data Loader, and Criterion Function

# Create Linear Regression Class
class linear_regression(nn.Module):

    # Constructor
    def __init__(self, input_size, output_size):
        super(linear_regression, self).__init__()
        self.linear = nn.Linear(input_size, output_size)

    # Prediction function
    def forward(self, x):
        yhat = self.linear(x)
        return yhat


# Create the criterion function and a DataLoader object:
criterion = nn.MSELoss()
trainloader = DataLoader(dataset=train_data)  # batch size is 1 by default


# #### Different learning rates and Data Structures to Store results for different Hyperparameters

# Create a list with different learning rates
learning_rates = [10**exp for exp in range(-4, 0)]


# #### Train different models  for different Hyperparameters

# Try different values of learning rates, perform stochastic gradient descent, and save the results on the training
# data and validation data. Finally, save each model in a list.
def train_model_with_lr(XY, val_XY, n_iter, lr_list):
    # take X and Y variables from Dataloader object
    X_tr, Y_tr = zip(*XY)
    X_tr, Y_tr = (torch.stack(X_tr).view(-1, 1), torch.stack(Y_tr).view(-1, 1))

    # create tensors (can be a lists) for the training and validating cost/total-loss. Include a list which stores the
    # training model for every value of the learning rate.
    train_err = torch.zeros(len(lr_list))
    valid_err = torch.zeros(len(lr_list))
    model_list = []

    # iterate through different learning rates
    for i, lr in enumerate(lr_list):
        model = linear_regression(1, 1)
        optimizer = optim.SGD(model.parameters(), lr=lr)
        # iterations for optimization of each model
        for epoch in range(n_iter):
            for x, y in XY:
                yhat = model(x)
                loss = criterion(yhat, y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        model_list.append(model)

        # train data
        train_loss = criterion(model(X_tr), Y_tr)
        train_err[i] = train_loss.item()

        # validation data
        val_loss = criterion(model(val_XY.x), val_XY.y)
        valid_err[i] = val_loss.item()

    return model_list, train_err, valid_err


MODELS, training_error, validation_error = train_model_with_lr(trainloader, val_data, 10, learning_rates)


# #### View the Results

# Plot the training loss and validation loss for each learning rate:
plt.figure()
plt.semilogx(np.array(learning_rates), training_error.numpy(), label='training cost')
plt.semilogx(np.array(learning_rates), validation_error.numpy(), label='validation cost')
# plt.plot(np.array(learning_rates), train_error.numpy(), label='training loss/total Loss')
# plt.plot(np.array(learning_rates), validation_error.numpy(), label='validation cost/total Loss')
plt.ylabel('Cost/Total-Loss')
plt.xlabel('Log Learning Rate')
plt.title("MSE with respect to the learning rate used\n for training on the train set")
plt.legend()
plt.show()

# Produce a prediction by using the validation data for each model:
plt.figure()
for model, lr in zip(MODELS, learning_rates):
    yhat = model(val_data.x)
    plt.plot(val_data.x.numpy(), yhat.detach().numpy(), label='LR = ' + str(lr))
plt.plot(val_data.x.numpy(), val_data.f.numpy(), 'og', label='validation data')
plt.plot(train_data.x.numpy(), train_data.f.numpy(), 'xg', label='training data')
plt.xlabel('x')
plt.ylabel('y')
plt.title("Predictions for different learning rates")
plt.legend()
plt.show()
