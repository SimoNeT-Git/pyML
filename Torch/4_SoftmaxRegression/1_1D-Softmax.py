#!/usr/bin/env python
# coding: utf-8

# In this lab, you will use Softmax to classify three linearly separable classes, the features are in one dimension
#   - Make Some Data
#   - Build Softmax Classifier
#   - Train the Model
#   - Analyze Results

import torch.nn as nn
import torch
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import Dataset, DataLoader


# Create class for plotting
def plot_data(data_set, model=None, n=1, color=False):
    X = data_set[:][0]
    Y = data_set[:][1]
    plt.figure()
    plt.plot(X[Y == 0, 0].numpy(), Y[Y == 0].numpy(), 'bo', label='y = 0')
    plt.plot(X[Y == 1, 0].numpy(), 0 * Y[Y == 1].numpy(), 'ro', label='y = 1')
    plt.plot(X[Y == 2, 0].numpy(), 0 * Y[Y == 2].numpy(), 'go', label='y = 2')
    plt.ylim((-0.1, 3))
    plt.xlabel('x')
    plt.xlabel('z')
    plt.legend()
    if model is not None:
        w = list(model.parameters())[0][0].detach()
        b = list(model.parameters())[1][0].detach()
        y_label = ['yhat=0', 'yhat=1', 'yhat=2']
        y_color = ['b', 'r', 'g']
        Y = []
        for w, b, y_l, y_c in zip(model.state_dict()['0.weight'], model.state_dict()['0.bias'], y_label, y_color):
            Y.append((w * X + b).numpy())
            plt.plot(X.numpy(), (w * X + b).numpy(), y_c, label=y_l)
        if color:
            x = X.numpy()
            x = x.reshape(-1)
            top = np.ones(x.shape)
            y0 = Y[0].reshape(-1)
            y1 = Y[1].reshape(-1)
            y2 = Y[2].reshape(-1)
            plt.fill_between(x, y0, where=y1 > y1, interpolate=True, color='blue')
            plt.fill_between(x, y0, where=y1 > y2, interpolate=True, color='blue')
            plt.fill_between(x, y1, where=y1 > y0, interpolate=True, color='red')
            plt.fill_between(x, y1, where=((y1 > y2) * (y1 > y0)), interpolate=True, color='red')
            plt.fill_between(x, y2, where=(y2 > y0) * (y0 > 0), interpolate=True, color='green')
            plt.fill_between(x, y2, where=(y2 > y1), interpolate=True, color='green')
    plt.legend()
    plt.show()


# Set the random seed:
torch.manual_seed(0)


# #### Make Some Data


# Create some linearly separable data with three classes:
class Data(Dataset):

    # Constructor
    def __init__(self):
        self.x = torch.arange(-2, 2, 0.1).view(-1, 1)
        self.y = torch.zeros(self.x.shape[0])
        self.y[(self.x > -1.0)[:, 0] * (self.x < 1.0)[:, 0]] = 1
        self.y[(self.x >= 1.0)[:, 0]] = 2
        self.y = self.y.type(torch.LongTensor)
        self.len = self.x.shape[0]

    # Getter
    def __getitem__(self, index):
        return self.x[index], self.y[index]

    # Get Length
    def __len__(self):
        return self.len


# Create the dataset object and plot it
data_set = Data()
plot_data(data_set)

# #### Build a Softmax Classifier

# Build a Softmax classifier by using the Sequential module (technically you only need nn.Linear)
model = nn.Sequential(nn.Linear(1, 3))
params = list(model.parameters())  # or params = list(model.state_dict().values())
print("\n\nThe weights of the Softmax classifier (that uses the torch.nn.linear model) "
      "randomly initialized are", params[0].tolist(), "\nwith shape", params[0].shape[0], "x", params[0].shape[1],
      "while the bias is", params[1].tolist())

# #### Train the Model

# Create the criterion function, the optimizer and the dataloader
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
trainloader = DataLoader(dataset=data_set, batch_size=5)

# Train the model for every 50 epochs plot, the line generated for each class.
LOSS = []


def train_model(epochs):
    for epoch in range(epochs):
        if epoch % 50 == 0:
            pass
            plot_data(data_set, model)
        for x, y in trainloader:
            optimizer.zero_grad()
            yhat = model(x)
            loss = criterion(yhat, y)
            LOSS.append(loss)
            loss.backward()
            optimizer.step()


train_model(300)

# #### Analyze Results

# Find the predicted class on the test data:
z = model(data_set.x)
_, yhat = z.max(1)
print("The actual labels are:", data_set.y.tolist(),
      "\nThe predictions are:  ", yhat.tolist())

# Calculate the accuracy on the test data:
correct = (data_set.y == yhat).sum().item()
accuracy = correct / len(data_set)
print("\nThe accuracy of the model is therefore:", accuracy * 100, "%")

# You can also use the softmax function to convert the output to a probability. First, we create a Softmax object:
Softmax_fn = nn.Softmax(dim=-1)

# The result is a tensor Probability, where each row corresponds to a different sample, and each column corresponds to
# that sample belonging to a particular class
Probability = Softmax_fn(z)

# we can obtain the probability of the first sample belonging to the first, second and third class respectively:
print("\nWe can also compute the probability for all samples of belonging to the different classes.\n"
      "E.g., for the first sample " + str(data_set.x[0].tolist()) + " we have:\n")
for i in range(3):
    print("Probability of class {} is {}".format(i, Probability[0, i]))
