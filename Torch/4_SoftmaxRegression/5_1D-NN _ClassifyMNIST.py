#!/usr/bin/env python
# coding: utf-8

# In this lab, you will use a single layer neural network to classify handwritten digits from the MNIST database.
# Note that as it is a multiclass classification problem we don't have the sigmoid activation function applied to the
# final (output) layer of the network!
#   - Neural Network Module and Training Function
#   - Make Some Data
#   - Define the Neural Network, Optimizer, and Train the  Model
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


# Use the following function for printing the model parameters:
def print_model_parameters(model):
    count = 0
    for ele in model.state_dict().keys():
        count += 1
        if count % 2 != 0:
            print("The following are the parameters for the layer ", count // 2 + 1)
        if ele.find("bias") != -1:
            print("The size of bias: ", list(model.state_dict()[ele].size()))
        else:
            print("The size of weights: ", list(model.state_dict()[ele].size()))


# Define a function to display data
def show_data(data_sample):
    plt.figure()
    plt.imshow(data_sample[0].numpy().reshape(28, 28), cmap='gray')
    plt.title('y = ' + str(data_sample[1]))  # label
    plt.show()


# #### Neural Network Module and Training Function

# Define the neural network module or class:
class Net(nn.Module):

    # Constructor
    def __init__(self, D_in, H, D_out):
        super(Net, self).__init__()
        self.linear1 = nn.Linear(D_in, H)
        self.linear2 = nn.Linear(H, D_out)

    # Prediction    
    def forward(self, x):
        x = torch.sigmoid(self.linear1(x))
        x = self.linear2(x)
        return x


# Define a function to train the model. In this case, the function returns a Python dictionary to store the training
# loss and accuracy on the validation data.
def train(train_loader, validation_loader, model, criterion, optimizer, epochs=100):
    useful_stuff = {'training_loss': [], 'training_cost': [], 'validation_accuracy': []}

    # training
    for epoch in range(epochs):
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
        acc = correct / (validation_loader.batch_size * len(validation_loader)) * 100
        useful_stuff['validation_accuracy'].append(acc)  # accuracy

    return useful_stuff


# #### Make Some Data

# Load the training dataset by setting the parameters train to True and convert it to a tensor by placing a transform
# object in the argument transform.
train_dataset = dsets.MNIST(root='./Data', train=True, download=True, transform=transforms.ToTensor())

# Load the testing dataset by setting the parameters train to False and convert it to a tensor by placing a transform
# object in the argument transform:
validation_dataset = dsets.MNIST(root='./Data', train=False, download=True, transform=transforms.ToTensor())

# Create criterion function
criterion = nn.CrossEntropyLoss()

# Create the training-data loader and the validation-data loader objects: 
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=2000, shuffle=True)
validation_loader = torch.utils.data.DataLoader(dataset=validation_dataset, batch_size=5000, shuffle=False)

# #### Define the Neural Network, Optimizer, and Train the Model

# Create the model with 100 neurons
input_dim = 28 * 28
hidden_dim = 100
output_dim = 10
model = Net(input_dim, hidden_dim, output_dim).to(device)

# Print the parameters for model
print_model_parameters(model)

# Define the optimizer object with a learning rate of 0.01:
learning_rate = 0.01
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# Train the model by using 100 epochs **(this process takes time)**:
epochs = 30
t0 = time()
training_results = train(train_loader, validation_loader, model, criterion, optimizer, epochs=epochs)
print("\nThe whole training process, which consists of", epochs, "epochs, took", round(time()-t0, 2), "seconds.")


# #### Analyze Results

# Plot the training total loss or cost for every iteration and plot the training accuracy for every epoch:
plot_accuracy_loss(training_results)

# Plot the first five misclassified samples:
Softmax_fn = nn.Softmax(dim=-1)
count = 0
for x, y in validation_dataset:
    z = model(x.to(device).reshape(-1, 28 * 28))
    _, yhat = torch.max(z, 1)
    if yhat != y:
        print("\nMisclassified sample number", count, "has actual label", y,
              "\nbut was attributed to class", yhat.item(), "with probability", torch.max(Softmax_fn(z)).item())
        show_data((x, y))
        count += 1
    if count >= 5:
        break

# Use nn.Sequential to build exactly the same model as you just built. Use the function train to train the model and use
# the function plot_accuracy_loss to see the metrics. Also, try different epoch numbers.
model_seq = torch.nn.Sequential(
    torch.nn.Linear(input_dim, hidden_dim),
    torch.nn.Sigmoid(),
    torch.nn.Linear(hidden_dim, output_dim),
).to(device)
optimizer = torch.optim.SGD(model_seq.parameters(), lr=learning_rate)
t0 = time()
training_results = train(train_loader, validation_loader, model_seq, criterion, optimizer, epochs=10)
print("\nThe whole training process with torch.nn.Sequential(), which consists of", epochs, "epochs, took",
      round(time()-t0, 2), "seconds.")
plot_accuracy_loss(training_results)
