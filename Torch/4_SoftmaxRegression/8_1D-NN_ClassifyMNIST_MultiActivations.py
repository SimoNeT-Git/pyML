#!/usr/bin/env python
# coding: utf-8

# In this lab, you will test sigmoid, tanh, and relu activation functions on the MNIST dataset.
#   - Neural Network Module and Training Function
#   - Make Some Data
#   - Define the common parameters and Criterion Function
#   - Train and Test Sigmoid, Tanh, and Relu
#   - Analyze Results

from time import time
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as dsets
import matplotlib.pylab as plt

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

# Define the neural network module or class using the sigmoid activation function:
class NetSig(nn.Module):

    # Constructor
    def __init__(self, D_in, H, D_out):
        super(NetSig, self).__init__()
        self.linear1 = nn.Linear(D_in, H)
        self.linear2 = nn.Linear(H, D_out)

    # Prediction
    def forward(self, x):
        x = torch.sigmoid(self.linear1(x))
        x = self.linear2(x)
        return x


# Define the neural network module or class using the Tanh activation function:
class NetTanh(nn.Module):

    # Constructor
    def __init__(self, D_in, H, D_out):
        super(NetTanh, self).__init__()
        self.linear1 = nn.Linear(D_in, H)
        self.linear2 = nn.Linear(H, D_out)

    # Prediction
    def forward(self, x):
        x = torch.tanh(self.linear1(x))
        x = self.linear2(x)
        return x


# Define the neural network module or class using the Relu activation function:
class NetRelu(nn.Module):

    # Constructor
    def __init__(self, D_in, H, D_out):
        super(NetRelu, self).__init__()
        self.linear1 = nn.Linear(D_in, H)
        self.linear2 = nn.Linear(H, D_out)

    # Prediction
    def forward(self, x):
        x = torch.relu(self.linear1(x))
        x = self.linear2(x)
        return x


# Define a function to train the model. In this case, the function returns a Python dictionary to store the training
# loss for each iteration  and accuracy on the validation data.
def train(train_loader, validation_loader, model, criterion, optimizer, epochs=100):
    useful_stuff = {'training_loss': [], 'training_cost': [], 'validation_accuracy': []}

    for epoch in range(epochs):
        # training
        tot_loss = 0
        for x, y in train_loader:
            optimizer.zero_grad()
            z = model(x.to(device).view(-1, 28 * 28))
            loss = criterion(z, y.to(device))
            loss.backward()
            optimizer.step()
            tot_loss += loss.data.item()
            useful_stuff['training_loss'].append(loss.data.item())  # loss at each iteration
        useful_stuff['training_cost'].append(tot_loss / (len(train_loader)))  # total loss
        # validation
        correct = 0
        for x_test, y_test in validation_loader:
            yhat = model(x_test.to(device).view(-1, 28 * 28))
            _, label = torch.max(yhat, 1)
            correct += (label == y_test.to(device)).sum().item()
        accuracy = correct / (validation_loader.batch_size * len(validation_loader)) * 100
        useful_stuff['validation_accuracy'].append(accuracy)

    return useful_stuff


# #### Make Some Data

# Load the training dataset by setting the parameters train to True and convert it to a tensor by placing a transform
# object in the argument transform.
train_dataset = dsets.MNIST(root='./Data', train=True, download=True, transform=transforms.ToTensor())

# Load the testing dataset by setting the parameter train to False and convert it to a tensor by placing a transform
# object in the argument transform.
validation_dataset = dsets.MNIST(root='./Data', train=False, download=True, transform=transforms.ToTensor())

# Create the training-data loader and the validation-data loader object:
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=2000, shuffle=True)
validation_loader = torch.utils.data.DataLoader(dataset=validation_dataset, batch_size=5000, shuffle=False)


# #### Define the common parameters and Criterion Function

# Create the criterion function:
criterion = nn.CrossEntropyLoss()

# Define common parameters for the models
input_dim = 28 * 28
hidden_dim = 100
output_dim = 10
learning_rate = 0.01
epochs = 30


# #### Train and test Sigmoid, Tanh, and Relu Neural Networks

# Train the network by using the sigmoid activations function:
model_Sig = NetSig(input_dim, hidden_dim, output_dim).to(device)
optimizer = torch.optim.SGD(model_Sig.parameters(), lr=learning_rate)
print("\nLets train the Neural Network with Sigmoid activation function.")
t0 = time()
training_results_sig = train(train_loader, validation_loader, model_Sig, criterion, optimizer, epochs=epochs)
print("The whole training process, which consists of", epochs, "epochs, took", round(time()-t0, 2), "seconds.")
plot_accuracy_loss(training_results_sig)

# Train the network by using the Tanh activations function:
model_Tanh = NetTanh(input_dim, hidden_dim, output_dim).to(device)
optimizer = torch.optim.SGD(model_Tanh.parameters(), lr=learning_rate)
print("\nLets train the Neural Network with Tanh activation function.")
t0 = time()
training_results_tanh = train(train_loader, validation_loader, model_Tanh, criterion, optimizer, epochs=epochs)
print("The whole training process, which consists of", epochs, "epochs, took", round(time()-t0, 2), "seconds.")
plot_accuracy_loss(training_results_tanh)

# Train the network by using the Relu activations function:
modelRelu = NetRelu(input_dim, hidden_dim, output_dim).to(device)
optimizer = torch.optim.SGD(modelRelu.parameters(), lr=learning_rate)
print("\nLets train the Neural Network with ReLu activation function.")
t0 = time()
training_results_relu = train(train_loader, validation_loader, modelRelu, criterion, optimizer, epochs=epochs)
print("The whole training process, which consists of", epochs, "epochs, took", round(time()-t0, 2), "seconds.")
plot_accuracy_loss(training_results_relu)


# #### Analyze Results

# Compare the training loss for each activation
plt.figure()
plt.plot(training_results_sig['training_loss'], label='sigmoid')
plt.plot(training_results_tanh['training_loss'], label='tanh')
plt.plot(training_results_relu['training_loss'], label='relu')
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.title('Loss on Training Set at each Iteration')
plt.legend()
plt.show()

# Print max accuracy of all models
print("\nLets look at the maximum accuracy value achieved with the 3 models tested.")
print("  - NN with Sigmoid activation function:", round(max(training_results_sig['validation_accuracy']), 2), "%")
print("  - NN with Tanh activation function:", round(max(training_results_tanh['validation_accuracy']), 2), "%")
print("  - NN with ReLu activation function:", round(max(training_results_relu['validation_accuracy']), 2), "%")

# Compare the validation loss for each model
plt.figure()
plt.plot(training_results_sig['validation_accuracy'], label='sigmoid')
plt.plot(training_results_tanh['validation_accuracy'], label='tanh')
plt.plot(training_results_relu['validation_accuracy'], label='relu')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.xlabel('Accuracy on Validation Set at each Epoch')
plt.legend()
plt.show()
