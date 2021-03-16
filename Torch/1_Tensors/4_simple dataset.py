#!/usr/bin/env python
# coding: utf-8

# In this lab, you will construct a basic dataset by using PyTorch and learn how to apply basic transformations to it.
#  - Simple dataset
#  - Transforms
#  - Compose

import torch
from torch.utils.data import Dataset
from torchvision import transforms

torch.manual_seed(1)


########################################### Simple Dataset
print("\n\nCreate Simple Dataset:")

# Let us try to create our own dataset class.
class toy_set(Dataset):

    # Constructor with default values: it makes an object containing 2 tensors (x and y), the first with shape 100 x 2
    # containing 100 copies of [2.,2.] and the second with shape 100 x 1 with 100 copies of [1.]
    # y with 1 element ([1])
    def __init__(self, length=100, transform=None):  # Note: we can pass a class to the transform attribute
        self.len = length
        self.x = 2 * torch.ones(length, 2)
        self.y = torch.ones(length, 1)
        self.transform = transform

    # Getter: it makes indexing of toy_set object possible, i.e. if we call toy_set[0], that equals
    # toy_set.__getitem__(0), it returns a tuple (x0, y0) = (tensor([2., 2.]), tensor([1.]))
    def __getitem__(self, index):
        sample = self.x[index], self.y[index]
        if self.transform:
            sample = self.transform(sample)  # if a class was passed it, returns the output of such transformation class
        return sample

    # Get Length: it returns the length of the tensors x and y, i.e. if we call len(toy_set), that equals
    # toy_set.__len__(), it returns 100
    def __len__(self):
        return self.len


# Now, let us create our toy_set object, and find out the value on index 1 and the length of the initial dataset
our_dataset = toy_set()
print("Our toy_set object: ", our_dataset)
print("Value on index 0 of our toy_set object: ", our_dataset[0])  # use __getitem__
print("Our toy_set length: ", len(our_dataset))  # use __len__
# As a result, we can apply the same indexing convention as a list, and apply the function len on the toy_set object.
# We are able to customize the indexing and length method by def __getitem__(self, index) and def __len__(self).


########################################### Transform Dataset
print("\n\nTransfrom Dataset:")

# You can also create a class for transforming the data. In this case, we will try to add 1 to x and multiply y by 2:
class add_mult(object):

    # Constructor: defines the scalar to add to x (default is addx=1) and the one to multiply to y (default is muly=2)
    def __init__(self, addx=1, muly=2):
        self.addx = addx
        self.muly = muly

    # Executor
    def __call__(self, sample):
        x = sample[0]
        y = sample[1]
        x = x + self.addx  # add
        y = y * self.muly  # multiply
        sample = x, y
        return sample


# Create an add_mult transform object, and a toy_set object
a_m = add_mult()
data_set = toy_set()

# Assign the outputs of the original dataset to x and y. Then, apply the transform add_mult to the dataset and output
# the values as x_ and y_, respectively:
# Use loop to print out first 5 elements in dataset
print("\nCreating a new transformation class:")
for i in range(5):
    x, y = data_set[i]
    print('Index: ', i, 'Original x: ', x, 'Original y: ', y)
    x_, y_ = a_m(data_set[i])
    print('Index: ', i, 'Transformed x_:', x_, 'Transformed y_:', y_)
# As a result, x has been added by 1 and y has been multiplied by 2, as [2, 2] + 1 = [3, 3] and [1] x 2 = [2]

# Note that we can apply the transform object every time we create a new toy_set object. Remember, we have the
# constructor in toy_set class with the parameter transform = None. When we create a new object using the constructor,
# we can assign the transform object to the parameter transform, as the following code demonstrates.
# Create a new data_set object with add_mult object as transform
cust_data_set = toy_set(transform=a_m)

# This applied a_m object (a transform method) to every element in cust_data_set as initialized. Let us print out the
# first 5 elements in cust_data_set in order to see whether the a_m applied on cust_data_set
print("\nApplying transformation directly to the toy_set object:")
for i in range(10):
    x, y = data_set[i]
    print('Index: ', i, 'Original x: ', x, 'Original y: ', y)
    x_, y_ = cust_data_set[i]
    print('Index: ', i, 'Transformed x_:', x_, 'Transformed y_:', y_)
# The result is the same as the previous method.


########################################### Compose Multiple Transforms on Dataset
print("\n\nCompose Multiple Transforms on Dataset:")

# You can compose multiple transforms on the dataset object. First, import transforms from torchvision:
# Create a new transform class that multiplies each of the elements by 100:
class mult(object):

    # Constructor
    def __init__(self, mult=100):
        self.mult = mult

    # Executor
    def __call__(self, sample):
        x = sample[0]
        y = sample[1]
        x = x * self.mult
        y = y * self.mult
        sample = x, y
        return sample


# Now let us try to combine the transforms add_mult and mult classes.
data_transform = transforms.Compose([add_mult(), mult()])
print("The combination of transforms (Compose): ", data_transform)

# The new Compose object will perform each transform concurrently.
data_set = toy_set()
x, y = data_set[0]
print('Original x: ', x, 'Original y: ', y)
x_, y_ = data_transform(data_set[0])
print('Transformed x_:', x_, 'Transformed y_:', y_)
# Let us see what happened on index 0. The original value of x is [2, 2], and the original value of y is [1]. If we only
# applied add_mult() on the original dataset, then the x becomes [3, 3] and y becomes [2]. Now let us see what is the
# value after applying both add_mult() and mult(). The result of x is [300, 300] and y is [200]. The calculation which
# is equivalent to the compose is x = ([2, 2] + 1) x 100 = [300, 300], y = ([1] x 2) x 100 = 200.

# Now we can pass the new Compose object (The combination of methods add_mult() and mult) to the constructor for
# creating toy_set object.
compose_data_set = toy_set(transform=transforms.Compose([add_mult(), mult()]))

# Let us print out the first 3 elements in different toy_set datasets in order to compare the output after different
# transforms have been applied:
data_set = toy_set()
cust_data_set = toy_set(transform=a_m)
for i in range(3):
    x, y = data_set[i]
    print('Index: ', i, 'Original x: ', x, 'Original y: ', y)
    x_, y_ = cust_data_set[i]
    print('Index: ', i, 'Add 1 to x:', x_, 'Multiply 2 to y', y_)
    x_co, y_co = compose_data_set[i]
    print('Index: ', i, 'Compose Transformed x_co: ', x_co, 'Compose Transformed y_co: ', y_co)
