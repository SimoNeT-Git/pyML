#!/usr/bin/env python
# coding: utf-8

# In this lab, you will see how adding dropout to your model will decrease overfitting.
#   - Make Some Data
#   - Create the Criterion object and set common parameters
#   - Create and train the models (with and without dropout) via Mini-Batch Gradient Descent
#   - Analyze results

from time import time
import torch
import matplotlib.pyplot as plt
import torch.nn as nn
import numpy as np
from matplotlib.colors import ListedColormap
from torch.utils.data import Dataset
torch.manual_seed(0)


# The function for plotting decision regions of classes
def plot_decision_regions_3class(data_set, model=None):
    cmap_light = ListedColormap(['#0000FF', '#FF0000'])
    # cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#00AAFF'])
    X = data_set.x.numpy()
    y = data_set.y.numpy()
    h = .02
    x_min, x_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
    y_min, y_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    newdata = np.c_[xx.ravel(), yy.ravel()]

    Z = data_set.multi_dim_poly(newdata).flatten()
    f = np.zeros(Z.shape)
    f[Z > 0] = 1
    f = f.reshape(xx.shape)
    plt.figure()
    if model is not None:
        model.eval()
        XX = torch.Tensor(newdata)
        _, yhat = torch.max(model(XX), 1)
        yhat = yhat.numpy().reshape(xx.shape)
        plt.pcolormesh(xx, yy, yhat, cmap=cmap_light)
        plt.contour(xx, yy, f, cmap=plt.cm.Paired)
        plt.title("Decision Region vs True Decision Boundary")
    else:
        plt.pcolormesh(xx, yy, f, cmap=cmap_light)
        plt.contour(xx, yy, f, cmap=plt.cm.Paired)
        plt.title("True Decision Regions and Boundary")
    plt.show()


# #### Make Some Data

# Create a non-linearly separable dataset:
class Data(Dataset):

    # Constructor
    def __init__(self, N_SAMPLES=1000, noise_std=0.15, train=True):
        a = np.matrix([-1, 1, 2, 1, 1, -3, 1]).T
        self.x = np.matrix(np.random.rand(N_SAMPLES, 2))
        self.f = np.array(
            a[0] + self.x * a[1:3] + np.multiply(self.x[:, 0], self.x[:, 1]) * a[4] + np.multiply(self.x, self.x) * a[
                                                                                                        5:7]).flatten()
        self.y = np.zeros(N_SAMPLES)
        self.y[self.f > 0] = 1
        self.y = torch.from_numpy(self.y).type(torch.LongTensor)
        self.x = torch.from_numpy(self.x).type(torch.FloatTensor)
        self.x = self.x + noise_std * torch.randn(self.x.size())
        self.f = torch.from_numpy(self.f)
        self.a = a
        if train:
            torch.manual_seed(1)
            self.x = self.x + noise_std * torch.randn(self.x.size())
            torch.manual_seed(0)

    # Getter        
    def __getitem__(self, index):
        return self.x[index], self.y[index]

    # Get Length
    def __len__(self):
        return self.len

    # Plot the diagram
    def plot(self):
        X = data_set.x.numpy()
        y = data_set.y.numpy()
        h = .02
        x_min, x_max = X[:, 0].min(), X[:, 0].max()
        y_min, y_max = X[:, 1].min(), X[:, 1].max()
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
        Z = data_set.multi_dim_poly(np.c_[xx.ravel(), yy.ravel()]).flatten()
        f = np.zeros(Z.shape)
        f[Z > 0] = 1
        f = f.reshape(xx.shape)
        plt.figure()
        plt.title('Noisy Sample Points and True Decision Boundary')
        plt.plot(self.x[self.y == 0, 0].numpy(), self.x[self.y == 0, 1].numpy(), 'bo', label='y=0')
        plt.plot(self.x[self.y == 1, 0].numpy(), self.x[self.y == 1, 1].numpy(), 'ro', label='y=1')
        plt.contour(xx, yy, f, cmap=plt.cm.Paired)
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.legend()
        plt.show()

    # Make a multidimension ploynomial function
    def multi_dim_poly(self, x):
        x = np.matrix(x)
        out = np.array(
            self.a[0] + x * self.a[1:3] + np.multiply(x[:, 0], x[:, 1]) * self.a[4] + np.multiply(x, x) * self.a[5:7])
        out = np.array(out)
        return out


# Create a dataset object and plot the dataset
data_set = Data(noise_std=0.2)
data_set.plot()

# Validation dataset object
validation_set = Data(train=False)

# Plot the correct decision boundary
plot_decision_regions_3class(data_set)

# #### Create the Criterion object and set common parameters

# Create a custom Net module with three layers. in_size is the size of the input features, n_hidden is the size of the
# layers, and out_size is the size. p is the dropout probability. The default is 0, that is, no dropout.

# Create the Cross Entropy Loss object
criterion = torch.nn.CrossEntropyLoss()

# Set common parameters
D_in = 2
H = 300
D_out = 2
lr = 0.01
epochs = 500


# #### Create and train the Models (with and without dropout) via Mini-Batch Gradient Descent

# Create a custom module with 2 hidden layers. in_size is the size of the input features, n_hidden is the size of the
# hidden layers, and out_size is the output size. p is dropout probability. The default is 0 which is no dropout.
class Net(nn.Module):

    # Constructor
    def __init__(self, in_size, n_hidden, out_size, p=0):
        super(Net, self).__init__()
        self.drop = nn.Dropout(p=p)
        self.linear1 = nn.Linear(in_size, n_hidden)
        self.linear2 = nn.Linear(n_hidden, n_hidden)
        self.linear3 = nn.Linear(n_hidden, out_size)

    # Prediction function
    def forward(self, x):
        x = torch.tanh(self.drop(self.linear1(x)))
        x = torch.tanh(self.drop(self.linear2(x)))
        x = self.linear3(x)
        return x


# Define function for training
def train(data, validation, modello, criterio, optim, n_epochs):
    # Initialize a dictionary that stores the training and validation loss
    LOSS = {'training_cost': [], 'validation_cost': [], 'validation_accuracy': []}
    for epoch in range(n_epochs):
        # all the samples are used for training
        yhat = modello(data.x)
        loss = criterio(yhat, data.y)

        # store the loss for both the training and validation data
        LOSS['training_cost'].append(loss.item())
        modello.eval()
        loss_val = criterio(modello(validation.x), validation.y)
        LOSS['validation_cost'].append(loss_val.item())
        LOSS['validation_accuracy'].append(accuracy(modello, validation))
        modello.train()

        # optimize model's parameters
        optim.zero_grad()
        loss.backward()
        optim.step()

    return LOSS


# The function for calculating accuracy
def accuracy(model, data_set):
    _, yhat = torch.max(model(data_set.x), 1)
    return (yhat == data_set.y).numpy().mean() * 100


## Without dropout
# Create and train the model with no dropout using batch gradient gradient descent
model = Net(D_in, H, D_out)
optimizer_ofit = torch.optim.Adam(model.parameters(), lr=lr)
print("\nLets train the Neural Network with no dropout.")
t0 = time()
LOSS = train(data_set, validation_set, model, criterion, optimizer_ofit, epochs)
print("The whole training process, which consists of", epochs, "epochs, took", round(time()-t0, 2), "seconds.")
plot_decision_regions_3class(data_set, model)

## With dropout at p=0.6
# Create and train the model with dropout using batch gradient gradient descent
model_drop = Net(D_in, H, D_out, p=0.5)
optimizer_drop = torch.optim.Adam(model_drop.parameters(), lr=lr)
# Set the model using dropout to training mode; this is the default mode, but it's good practice to write this
# in your code
model_drop.train()
print("\nLets train the Neural Network with dropout.")
t0 = time()
LOSS_drop = train(data_set, validation_set, model_drop, criterion, optimizer_drop, epochs)
print("The whole training process, which consists of", epochs, "epochs, took", round(time()-t0, 2), "seconds.")
# Set the model with dropout to evaluation mode after training is completed
model_drop.eval()
plot_decision_regions_3class(data_set, model_drop)
# You can see that the model using dropout does better at tracking the function that generated the data.


# #### Analyze results

# Plot out the loss for training and validation data on both models. We use the log to make the difference more apparent
dict_plot = {'training data no dropout': LOSS['training_cost'],
             'validation data no dropout': LOSS['validation_cost'],
             'training data dropout': LOSS_drop['training_cost'],
             'validation data dropout': LOSS_drop['validation_cost']}
print("\nYou see that the model without dropout performs better on the training data, but it performs worse on",
      "\nthe validation data. This suggests overfitting. Instead, the model using dropout performed better on",
      "\nthe validation data, but worse on the training data, thus avoiding overfitting.")
plt.figure(figsize=(6.1, 10))
for key, value in dict_plot.items():
    plt.plot(np.log(np.array(value)), label=key)
plt.legend()
plt.xlabel("Epoch")
plt.ylabel("Log of Cost")
plt.title('Comparison of Total Loss')
plt.show()

# Print last accuracy value of both models
print("\nThe accuracy of the model without dropout at the last epoch is:",
      round(LOSS['validation_accuracy'][-1], 1), "%")
print("The accuracy of the model with dropout at the last epoch is:",
      round(LOSS_drop['validation_accuracy'][-1], 1), "%")

# Plot accuracy
plt.figure()
plt.plot(LOSS['validation_accuracy'], label='no dropout')
plt.plot(LOSS_drop['validation_accuracy'], label='dropout')
plt.legend()
plt.xlabel('Epoch')
plt.ylabel("Accuracy")
plt.title('Comparison of accuracy on validation data')
plt.show()
# You see that the model with dropout performs better on the validation data.
