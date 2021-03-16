#!/usr/bin/env python
# coding: utf-8

# In this lab, you will test PyTroch Default Initialization, Xavier Initialization (only for Tanh activation) and
# Uniform Initialization on the MNIST dataset.
#   - Neural Network Module and Training Function
#   - Make Some Data
#   - Define Several Neural Network, Criterion function, Optimizer
#   - Test Uniform, Default and Xavier Initialization
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
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    ax1.plot(training_results['training_loss'], 'r')
    ax1.set_ylabel('Loss')
    ax1.set_xlabel('Iteration')
    ax1.set_title('Loss on Training Set at each Iteration')

    color = 'tab:red'
    ax2.plot(training_results['training_cost'], color=color)
    ax2.set_ylabel('Total Loss on Training Set', color=color)
    ax2.set_xlabel('Epoch')
    ax2.tick_params(axis='y', color=color)
    ax2.set_title('Cost / Accuracy after every Epoch')

    ax3 = ax2.twinx()
    color = 'tab:blue'
    ax3.plot(training_results['validation_accuracy'], color=color)
    ax3.set_ylabel('Accuracy on Validation Set', color=color)
    ax3.tick_params(axis='y', color=color)
    fig.tight_layout()

    plt.show()


# #### Neural Network Module and Training Function

# Define the neural network module or class with Xavier Initialization
class Net_Xavier(nn.Module):

    # Constructor
    def __init__(self, Layers):
        super(Net_Xavier, self).__init__()
        self.hidden = nn.ModuleList()

        for input_size, output_size in zip(Layers, Layers[1:]):
            linear = nn.Linear(input_size, output_size)
            torch.nn.init.xavier_uniform_(linear.weight)
            self.hidden.append(linear)

    # Prediction
    def forward(self, x):
        L = len(self.hidden)
        for (l, linear_transform) in zip(range(L), self.hidden):
            if l < L - 1:
                x = torch.tanh(linear_transform(x))
            else:
                x = linear_transform(x)
        return x


# Define the neural network module with Uniform Initialization:
class Net_Uniform(nn.Module):

    # Constructor
    def __init__(self, Layers):
        super(Net_Uniform, self).__init__()
        self.hidden = nn.ModuleList()

        for input_size, output_size in zip(Layers, Layers[1:]):
            linear = nn.Linear(input_size, output_size)
            linear.weight.data.uniform_(0, 1)
            self.hidden.append(linear)

    # Prediction
    def forward(self, x):
        L = len(self.hidden)
        for (l, linear_transform) in zip(range(L), self.hidden):
            if l < L - 1:
                x = torch.tanh(linear_transform(x))
            else:
                x = linear_transform(x)
        return x


# Define the neural network module with PyTorch Default Initialization
class Net_default(nn.Module):

    # Constructor
    def __init__(self, Layers):
        super(Net_default, self).__init__()
        self.hidden = nn.ModuleList()

        for input_size, output_size in zip(Layers, Layers[1:]):
            linear = nn.Linear(input_size, output_size)
            self.hidden.append(linear)

    # Prediction
    def forward(self, x):
        L = len(self.hidden)
        for (l, linear_transform) in zip(range(L), self.hidden):
            if l < L - 1:
                x = torch.tanh(linear_transform(x))
            else:
                x = linear_transform(x)
        return x


# Define a function to train the model, in this case the function returns a Python dictionary to store the training loss
# and accuracy on the validation data
def train(train_loader, validation_loader, model, criterion, optimizer, epochs=100):
    loss_accuracy = {'training_loss': [], 'training_cost': [], 'validation_accuracy': []}
    for epoch in range(epochs):

        tot_loss = 0
        for x, y in train_loader:
            optimizer.zero_grad()
            z = model(x.to(device).view(-1, 28 * 28))
            loss = criterion(z, y.to(device))
            loss.backward()
            optimizer.step()
            tot_loss += loss.item()
            loss_accuracy['training_loss'].append(loss.data.item())
        loss_accuracy['training_cost'].append(tot_loss / len(train_loader))

        correct = 0
        for x, y in validation_loader:
            yhat = model(x.to(device).view(-1, 28 * 28))
            _, label = torch.max(yhat, 1)
            correct += (label == y.to(device)).sum().item()
        accuracy = 100 * (correct / (validation_loader.batch_size * len(validation_loader)))
        loss_accuracy['validation_accuracy'].append(accuracy)

    return loss_accuracy


# #### Make Some Data

# Load the training dataset by setting the parameters train to True and convert it to a tensor by placing a transform
# object int the argument <code>transform</code>
train_dataset = dsets.MNIST(root='./Data', train=True, download=True, transform=transforms.ToTensor())

# Load the testing dataset by setting the parameters train to False and convert it to a tensor by placing a transform
# object int the argument transform
validation_dataset = dsets.MNIST(root='./Data', train=False, download=True, transform=transforms.ToTensor())

# Create the training-data loader and the validation-data loader object
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=2000, shuffle=True)
validation_loader = torch.utils.data.DataLoader(dataset=validation_dataset, batch_size=5000, shuffle=False)


# #### Define Neural Network, Criterion function, Optimizer and Train the Model

# Create the criterion function
criterion = nn.CrossEntropyLoss()

# Set the parameters
input_dim = 28 * 28
output_dim = 10
layers = [input_dim, 100, 10, 100, 10, 100, output_dim]
epochs = 15
learning_rate = 0.01


# #### Test PyTorch Default Initialization, Xavier Initialization, Uniform Initialization

# Train the network using PyTorch Default Initialization
print('\nLets create and train the NN with Default initialization.')
model_default = Net_default(layers).to(device)
optimizer_default = torch.optim.SGD(model_default.parameters(), lr=learning_rate)
print("Default initial parameters of the model are:")
for k, v in zip(list(model_default.state_dict().keys())[2:], list(model_default.state_dict().values())[2:]):
    print(k, ":  ", v.tolist())
t0 = time()
training_results_default = train(train_loader, validation_loader, model_default, criterion, optimizer_default, epochs=epochs)
print("The whole training process, which consists of", epochs, "epochs, took", round(time()-t0, 2), "seconds.")
plot_accuracy_loss(training_results_default)

# Train the network using Xavier Initialization function
print('\nLets create and train the NN with Xavier initialization method.')
model_Xavier = Net_Xavier(layers).to(device)
optimizer_Xavier = torch.optim.SGD(model_Xavier.parameters(), lr=learning_rate)
print("Xavier initial parameters of the model are:")
for k, v in zip(list(model_Xavier.state_dict().keys())[2:], list(model_Xavier.state_dict().values())[2:]):
    print(k, ":  ", v.tolist())
t0 = time()
training_results_Xavier = train(train_loader, validation_loader, model_Xavier, criterion, optimizer_Xavier, epochs=epochs)
print("The whole training process, which consists of", epochs, "epochs, took", round(time()-t0, 2), "seconds.")
plot_accuracy_loss(training_results_Xavier)

# Train the network using Uniform Initialization
print('\nLets create and train the NN with Uniform initialization method.')
model_Uniform = Net_Uniform(layers).to(device)
optimizer_Uniform = torch.optim.SGD(model_Uniform.parameters(), lr=learning_rate)
print("Uniform initial parameters of the model are:")
for k, v in zip(list(model_Uniform.state_dict().keys())[2:], list(model_Uniform.state_dict().values())[2:]):
    print(k, ":  ", v.tolist())
t0 = time()
training_results_Uniform = train(train_loader, validation_loader, model_Uniform, criterion, optimizer_Uniform, epochs=epochs)
print("The whole training process, which consists of", epochs, "epochs, took", round(time()-t0, 2), "seconds.")
plot_accuracy_loss(training_results_Uniform)


# #### Analyze Results

# Compare the training loss and validation accuracy for each initialization
fig = plt.figure(figsize=(10, 6))
plt.subplot(121)
plt.plot(training_results_default['training_loss'], label='Default')
plt.plot(training_results_Xavier['training_loss'], label='Xavier')
plt.plot(training_results_Uniform['training_loss'], label='Uniform')
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.title('Loss on Training Set at each Iteration', fontsize=10)
plt.legend()
plt.subplot(122)
plt.plot(training_results_default['validation_accuracy'], label='Default')
plt.plot(training_results_Xavier['validation_accuracy'], label='Xavier')
plt.plot(training_results_Uniform['validation_accuracy'], label='Uniform')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Accuracy on Validation Set at each Epoch', fontsize=10)
plt.legend()
fig.suptitle('Comparison of different initialization methods')
plt.show()
