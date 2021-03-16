#!/usr/bin/env python
# coding: utf-8

# In this lab, you will see how adding dropout to your model will decrease overfitting.
#   - Make Some Data
#   - Create the Criterion object and set common parameters
#   - Create and train the Models (with and without dropout) via Mini-Batch Gradient Descent
#   - Analyze results

from time import time
import torch
import matplotlib.pyplot as plt
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset
torch.manual_seed(0)


# #### Make Some Data

# Create polynomial dataset class:
class Data(Dataset):

    # Constructor
    def __init__(self, N_SAMPLES=40, noise_std=1, train=True):
        self.x = torch.linspace(-1, 1, N_SAMPLES).view(-1, 1)
        self.f = self.x ** 2
        if train:
            self.y = self.f + noise_std * torch.randn(self.f.size())
            self.y = self.y.view(-1, 1)
        else:
            torch.manual_seed(1)
            self.y = self.f + noise_std * torch.randn(self.f.size())
            self.y = self.y.view(-1, 1)
            torch.manual_seed(0)

    # Getter
    def __getitem__(self, index):
        return self.x[index], self.y[index]

    # Get Length
    def __len__(self):
        return self.len

    # Plot the data
    def plot(self):
        plt.figure()
        plt.scatter(self.x.numpy(), self.y.numpy(), label="Samples")
        plt.plot(self.x.numpy(), self.f.numpy(), label="True Function", color='orange')
        plt.xlabel("x")
        plt.ylabel("y")
        plt.xlim((-1, 1))
        plt.ylim((-2, 2.5))
        plt.title('Data Points and Best Fit')
        plt.legend(loc="best")
        plt.show()


# Create the dataset object and plot the dataset
data_set = Data()
data_set.plot()

# Create validation dataset object
validation_set = Data(train=False)


# #### Create the Criterion object and set common parameters

# Create the Mean Squared Error Loss object
criterion = torch.nn.MSELoss()

# Set common parameters
D_in = 1
H = 300
D_out = 1
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
        x = torch.relu(self.drop(self.linear1(x)))
        x = torch.relu(self.drop(self.linear2(x)))
        x = self.linear3(x)
        return x


# Define function for training
def train(data, validation, modello, criterio, optim, n_epochs):
    # Initialize a dictionary that stores the training and validation loss
    LOSS = {'training_cost': [], 'validation_cost': []}
    for epoch in range(n_epochs):
        # all the samples are used for training
        yhat = modello(data.x)
        loss = criterio(yhat, data.y)

        # store the loss for both the training and validation data
        LOSS['training_cost'].append(loss.item())
        modello.eval()
        loss_val = criterio(modello(validation.x), validation.y)
        LOSS['validation_cost'].append(loss_val.item())
        modello.train()

        # optimize model's parameters
        optim.zero_grad()
        loss.backward()
        optim.step()

    return LOSS


## Without dropout
# Create and train the model with no dropout using batch gradient gradient descent
model = Net(D_in, H, D_out)
optimizer_ofit = torch.optim.Adam(model.parameters(), lr=lr)
print("\nLets train the Neural Network with no dropout.")
t0 = time()
LOSS = train(data_set, validation_set, model, criterion, optimizer_ofit, epochs)
print("The whole training process, which consists of", epochs, "epochs, took", round(time()-t0, 2), "seconds.")
# Make a prediction
yhat = model(data_set.x)

## With dropout at p=0.5
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
# Set the model with dropout to evaluation mode after training is completed. Now we can make a prediction.
model_drop.eval()
yhat_drop = model_drop(data_set.x)


# #### Analyze results

# Plot predictions of both models. Compare them to the training points and the true function:
plt.figure()
plt.scatter(data_set.x.numpy(), data_set.y.numpy(), label="Samples")
plt.plot(data_set.x.numpy(), data_set.f.numpy(), label="True function", color='orange')
plt.plot(data_set.x.numpy(), yhat.detach().numpy(), label='No Dropout', c='r')
plt.plot(data_set.x.numpy(), yhat_drop.detach().numpy(), label="Dropout", c='g')
plt.xlabel("x")
plt.ylabel("y")
plt.xlim((-1, 1))
plt.ylim((-2, 2.5))
plt.legend(loc="best")
plt.show()
# You can see that the model using dropout does better at tracking the function that generated the data.

# Plot out the loss for training and validation data on both models. We use the log to make the difference more apparent
dict_plot = {'training data no dropout': LOSS['training_cost'],
             'validation data no dropout': LOSS['validation_cost'],
             'training data dropout': LOSS_drop['training_cost'],
             'validation data dropout': LOSS_drop['validation_cost']}
print("\nYou see that the model without dropout performs better on the training data, but it performs worse on",
      "\nthe validation data. This suggests overfitting. Instead, the model using dropout performed almost the same on",
      "\ntraining and validation data, thus avoiding overfitting.")
plt.figure(figsize=(6.1, 10))
for key, value in dict_plot.items():
    plt.plot(np.log(np.array(value)), label=key)
plt.legend()
plt.xlabel("Epoch")
plt.ylabel("Log of Cost")
plt.title('Comparison of Total Loss')
plt.show()

# Print last accuracy value of both models
print("\nThe total loss on validation set of the model without dropout at the last epoch is:",
      round(LOSS['validation_cost'][-1], 2))
print("The total loss on validation set of the model with dropout at the last epoch is:",
      round(LOSS_drop['validation_cost'][-1], 2))
# You see that the model with dropout performs better on the validation data.

# Plot total loss
plt.figure()
plt.plot(LOSS['validation_cost'], label='no dropout')
plt.plot(LOSS_drop['validation_cost'], label='dropout')
plt.legend()
plt.xlabel('Epoch')
plt.ylabel("Cost")
plt.title('Comparison of Total Loss on Validation Data')
plt.show()
