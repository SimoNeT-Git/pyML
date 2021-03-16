#!/usr/bin/env python
# coding: utf-8

# In this lab, you will build a Neural Network using Batch Normalization and compare it to a Neural Network that does
# not use Batch Normalization. You will use the MNIST dataset to test your network.
#   - Neural Network Modules and Training Function
#   - Load Data
#   - Define Criterion function and training parameters
#   - Train Neural Network using Batch Normalization and no Batch Normalization
#   - Analyze Results

from time import time
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as dsets
import matplotlib.pylab as plt

torch.manual_seed(0)

# Use GPU if possible! Note: for small models like this one there will be no speed-up of the training process.
use_cuda = torch.cuda.is_available()
if use_cuda:  # Print additional info when using cuda
    device = torch.device('cuda')
    current_dev = torch.cuda.current_device()
    cap_dev = torch.cuda.get_device_capability()
    print('\nRunning on GPU', torch.cuda.get_device_name(device),
          '(compute capability {}.{})'.format(cap_dev[0], cap_dev[1]),
          'number', current_dev + 1, 'of', torch.cuda.device_count())
    print('Total memory:     ', round(torch.cuda.get_device_properties(current_dev).total_memory / 1024 ** 2, 1), 'MB')
    print('Allocated memory: ', round(torch.cuda.memory_allocated(current_dev) / 1024 ** 2, 1), 'MB')
    print('Cached memory:    ', round(torch.cuda.memory_cached(current_dev) / 1024 ** 2, 1), 'MB\n')
else:
    device = torch.device('cpu')
    print('\nRunning on CPU.\n')


# Use the following helper functions for plotting the loss:
def plot_accuracy_loss(training_results):

    fig = plt.figure(figsize=(10, 6))
    plt.subplot(121)
    plt.plot(training_results['training_loss'])
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title('Loss on Training Set at every Iteration', fontsize=10)
    plt.subplot(122)
    plt.plot(training_results['validation_accuracy'])
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Accuracy on Validation Set at every Epoch', fontsize=10)
    fig.suptitle('Loss and Accuracy')
    plt.show()


# #### Neural Network Modules and Training Function

# Define the Neural Network Model with two hidden layers using Batch Normalization
class NetBatchNorm(nn.Module):

    # Constructor
    def __init__(self, in_size, n_hidden1, n_hidden2, out_size):
        super(NetBatchNorm, self).__init__()
        self.linear1 = nn.Linear(in_size, n_hidden1)
        self.linear2 = nn.Linear(n_hidden1, n_hidden2)
        self.linear3 = nn.Linear(n_hidden2, out_size)
        self.bn1 = nn.BatchNorm1d(n_hidden1)
        self.bn2 = nn.BatchNorm1d(n_hidden2)

    # Prediction
    def forward(self, x):
        x = self.bn1(torch.sigmoid(self.linear1(x)))
        x = self.bn2(torch.sigmoid(self.linear2(x)))
        x = self.linear3(x)
        return x

    # Activations, to analyze results 
    def activation(self, x):
        out = []
        z1 = self.bn1(self.linear1(x))
        out.append(z1.detach().cpu().numpy().reshape(-1))
        a1 = torch.sigmoid(z1)
        out.append(a1.detach().cpu().numpy().reshape(-1))
        z2 = self.bn2(self.linear2(a1))
        out.append(z2.detach().cpu().numpy().reshape(-1))
        a2 = torch.sigmoid(z2)
        out.append(a2.detach().cpu().numpy().reshape(-1))
        return out


# Define the Neural Network Model with two hidden layers without Batch Normalization
class Net(nn.Module):

    # Constructor
    def __init__(self, in_size, n_hidden1, n_hidden2, out_size):
        super(Net, self).__init__()
        self.linear1 = nn.Linear(in_size, n_hidden1)
        self.linear2 = nn.Linear(n_hidden1, n_hidden2)
        self.linear3 = nn.Linear(n_hidden2, out_size)

    # Prediction
    def forward(self, x):
        x = torch.sigmoid(self.linear1(x))
        x = torch.sigmoid(self.linear2(x))
        x = self.linear3(x)
        return x

    # Activations, to analyze results 
    def activation(self, x):
        out = []
        z1 = self.linear1(x)
        out.append(z1.detach().cpu().numpy().reshape(-1))
        a1 = torch.sigmoid(z1)
        out.append(a1.detach().cpu().numpy().reshape(-1).reshape(-1))
        z2 = self.linear2(a1)
        out.append(z2.detach().cpu().numpy().reshape(-1))
        a2 = torch.sigmoid(z2)
        out.append(a2.detach().cpu().numpy().reshape(-1))
        return out


# Define a function to train the model. In this case the function returns a Python dictionary to store the training loss
# and accuracy on the validation data
def train(train_loader, validation_loader, model, criterion, optimizer, epochs=100):
    useful_stuff = {'training_loss': [], 'validation_accuracy': []}

    for epoch in range(epochs):
        for x, y in train_loader:
            model.train()
            optimizer.zero_grad()
            z = model(x.to(device).view(-1, 28 * 28))
            loss = criterion(z, y.to(device))
            loss.backward()
            optimizer.step()
            useful_stuff['training_loss'].append(loss.data.item())

        correct = 0
        for x, y in validation_loader:
            model.eval()
            yhat = model(x.to(device).view(-1, 28 * 28))
            _, label = torch.max(yhat, 1)
            correct += (label == y.to(device)).sum().item()
        accuracy = 100 * correct / (validation_loader.batch_size * len(validation_loader))
        useful_stuff['validation_accuracy'].append(accuracy)

    return useful_stuff


# #### Make Some Data

# Load the training dataset by setting the parameters train to True and convert it to a tensor by placing a transform
# object int the argument transform.
train_dataset = dsets.MNIST(root='./Data', train=True, download=True, transform=transforms.ToTensor())

# Load the validating dataset by setting the parameters train False and convert it to a tensor by placing a transform
# object into the argument transform.
validation_dataset = dsets.MNIST(root='./Data', train=False, download=True, transform=transforms.ToTensor())

# Create the training data loader and the validation data loader object.
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=2000, shuffle=True)
validation_loader = torch.utils.data.DataLoader(dataset=validation_dataset, batch_size=5000, shuffle=False)


# #### Define Criterion function and training parameters

# Create the criterion function
criterion = nn.CrossEntropyLoss()

# Set the variables for Neural Network Shape hidden_dim used for number of neurons in both hidden layers.
input_dim = 28 * 28
hidden_dim = 100
output_dim = 10
learning_rate = 0.1
epochs = 10


# #### Train Neural Network using Batch Normalization and no Batch Normalization

# Create model without Batch Normalization, create optimizer and train the model
model = Net(input_dim, hidden_dim, hidden_dim, output_dim).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
print("\nLets train the Neural Network without Batch Normalization.")
t0 = time()
training_results = train(train_loader, validation_loader, model, criterion, optimizer, epochs=epochs)
print("The whole training process, which consists of", epochs, "epochs, took", round(time()-t0, 2), "seconds.")
plot_accuracy_loss(training_results)

# Create model using Batch Normalization, create optimizer and train the model
model_norm = NetBatchNorm(input_dim, hidden_dim, hidden_dim, output_dim).to(device)
optimizer = torch.optim.Adam(model_norm.parameters(), lr=learning_rate)
print("\nLets train the Neural Network with Batch Normalization.")
t0 = time()
training_results_Norm = train(train_loader, validation_loader, model_norm, criterion, optimizer, epochs=epochs)
print("The whole training process, which consists of", epochs, "epochs, took", round(time()-t0, 2), "seconds.")
plot_accuracy_loss(training_results_Norm)


# #### Analyze Results

# Compare the histograms of the activation for the first layer of the first sample, for both models.
model.eval()
out = model.activation(validation_dataset[0][0].to(device).reshape(-1, 28 * 28))
model_norm.eval()
out_norm = model_norm.activation(validation_dataset[0][0].to(device).reshape(-1, 28 * 28))
plt.figure()
plt.hist(out[2], label='without Batch Normalization')
plt.hist(out_norm[2], label='with Batch Normalization')
plt.xlabel("Activation")
plt.title('Comparison of activations for first sample in first layer of both models')
plt.legend()
plt.show()
# We see the activations with Batch Normalization are zero centred and have a smaller variance.

# Compare the training loss for each iteration
fig = plt.figure(figsize=(10, 6))
plt.subplot(121)
plt.plot(training_results['training_loss'], label='without Batch Normalization')
plt.plot(training_results_Norm['training_loss'], label='with Batch Normalization')
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.title('Loss on Training Set at each Iteration')
plt.legend()
plt.subplot(122)
plt.plot(training_results['validation_accuracy'], label='without Batch Normalization')
plt.plot(training_results_Norm['validation_accuracy'], label='with Batch Normalization')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Accuracy on Validation Set at each Epoch')
plt.legend()
fig.suptitle('Comparison of Loss and Accuracy')
plt.show()
