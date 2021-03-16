#!/usr/bin/env python
# coding: utf-8

# In this lab, you will learn the basics of tensor operations. Tensors are an essential part of PyTorch; there are
# complex mathematical objects in and of themselves. Fortunately, most of the intricacies are not necessary. In this
# section, you will compare them to vectors and numpy arrays.
#   - Types and Shape
#   - Indexing and Slicing
#   - Tensor Functions
#   - Tensor Operations
#   - Device Operations

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# This is the function for plotting diagrams. You will use this function to plot the vectors in Coordinate system.
# Plot vecotrs, please keep the parameters in the same length
# @param: Vectors = [{"vector": vector variable, "name": name of vector, "color": color of the vector on diagram}]

def plotVec(vectors):
    ax = plt.axes()

    # For loop to draw the vectors
    for vec in vectors:
        ax.arrow(0, 0, *vec["vector"], head_width=0.05, color=vec["color"], head_length=0.1)
        plt.text(*(vec["vector"] + 0.1), vec["name"])

    plt.ylim(-2, 2)
    plt.xlim(-2, 2)


########################################### Data type
print('\n\n--> DATA TYPE:')

# You can find the type of the following list of integers [0, 1, 2, 3, 4] by applying the constructor torch.tensor():
list_ints = [0, 1, 2, 3, 4]
ints_tensor = torch.tensor(list_ints)  # Convert an integer list with length 5 to a tensor
print("The data type of tensor object with int list is", ints_tensor.dtype,
      "\nwhile the tensor type is", ints_tensor.type(), "which is still a", type(ints_tensor))

# You can find the type of this float list [0.0, 1.0, 2.0, 3.0, 4.0] by applying the method torch.tensor():
list_floats = [0.0, 1.0, 2.0, 3.0, 4.0]
floats_tensor = torch.tensor(list_floats)  # Convert a float list with length 5 to a tensor
print("\nThe data type of tensor object with float list is", floats_tensor.dtype,
      "\nwhile the tensor type is", floats_tensor.type(), "which is still a", type(floats_tensor))

# The float list can also be converted to an INT tensor.
floats_int_tensor = torch.tensor(list_floats, dtype=torch.int64)
print("\nThe data type of tensor object with float list and dtype specified as argument is now",
      floats_int_tensor.dtype, "\nwhile the tensor type is", floats_int_tensor.type(),
      "which is still a", type(floats_int_tensor))
# Note: The elements in the list that will be converted to tensor must have the same type.

########################################### Tensor type
print('\n\n--> TENSOR TYPE:')

# From the previous examples, you see that torch.tensor() converts the list to the tensor type, which is similar to the
# original list type. However, what if you want to convert the list to a certain tensor type? torch contains the methods
# required to do this conversion. The following code converts an integer list to float tensor:
new_float_tensor = torch.FloatTensor(list_ints)  # Convert an integer list with length 5 to float tensor
print("\nThe tensor type of the tensor object with int list and tensor type specified is", new_float_tensor.type())

# Another method to convert the integer list to float tensor: you can convert an existing tensor object to another
# tensor type.
old_int_tensor = torch.tensor(list_ints)
new_float_tensor = old_int_tensor.type(torch.FloatTensor)  # Convert the integer tensor to a float tensor
print("\nUsing another method, the tensor type of the tensor object with int list and tensor type specified is",
      new_float_tensor.type())

###########################################  Tensor size/shape
print('\n\n--> TENSOR SIZE:')

# The tensor_obj.size() method helps you to find out the size of the tensor_obj.
# The tensor_obj.ndimension() method shows the dimension of the tensor object.
print("The size of the last tensor that was made is", new_float_tensor.size())
print("Its dimension is", new_float_tensor.ndimension())

########################################### Reshape Tensor
print('\n\n--> RESHAPE TENSOR:')

# The tensor_obj.view(row, column) is used for reshaping a tensor object.
# What if you have a tensor object with torch.Size([5]) as a new_float_tensor as shown in the previous example?
# After you execute new_float_tensor.view(5, 1), the size of new_float_tensor will be torch.Size([5, 1]).
# This means that the tensor object new_float_tensor has been reshaped from a 1D tensor object with 5 elements to a
# 2D tensor object with 5 rows and 1 column.
twoD_float_tensor = new_float_tensor.view(5, 1)
print("Original Size: ", new_float_tensor.shape)
print("Size after view method", twoD_float_tensor.shape)

# The original size is 5. The tensor after reshaping becomes a 5x1 tensor analog to a column vector.
# Note: The number of elements in a tensor must remain constant after applying view.

########################################### Reshape Tensor with dynamic size
print('\n\n--> RESHAPE TENSOR WITH DYNAMIC SIZE:')

# What if you have a tensor with dynamic size but you want to reshape it? You can use -1 to do just that.
twoD_float_tensor = new_float_tensor.view(-1, 1)
print("Original Size: ", new_float_tensor.shape)
print("Size after view method", twoD_float_tensor.shape)

# You get the same result as the previous example. The -1 can represent any size. However, be careful because you can
# set only one argument as -1.

########################################### Numpy to Tensor
print('\n\n--> FROM NUMPY TO TENSOR:')

# You can also convert a numpy array to a tensor, for example:
numpy_array = np.array(list_floats)  # Convert a numpy array to a tensor
new_tensor = torch.from_numpy(numpy_array)
print("The data type of tensor achieved from numpy array of floats is", new_tensor.dtype)
print("The tensor type is", new_tensor.type())

########################################### Tensor to Numpy
print('\n\n--> FROM TENSOR TO NUMPY:')

# Converting a tensor to a numpy is also supported in PyTorch. The syntax is shown below:
back_to_numpy = new_tensor.numpy()  # Convert a tensor to a numpy array
print("The data type of numpy array achieved from tensor of floats is", back_to_numpy.dtype)

# back_to_numpy and new_tensor still point to numpy_array. As a result if we change numpy_array both back_to_numpy and
# new_tensor will change. For example if we set all the elements in numpy_array to zeros, back_to_numpy and new_tensor
# will follow suit.
numpy_array[:] = 0  # Set all elements in numpy array to zero
print("If we set all elements of numpy_array to 0, the tensor achieved from it will also be of zeros: ", new_tensor)
print("And back_to_numpy array will follow: ", back_to_numpy)

########################################### From Pandas Series to Tensor
print('\n\n--> FROM PANDAS SERIES TO TENSOR AND LIST:')

# Pandas Series can also be converted by using the numpy array that is stored in pandas_series.values. Note that pandas
# series can be any pandas_series object.
list_rand = [0.1, 2, 0.3, 10.1]
pandas_series = pd.Series(list_rand)  # Create a panda series
new_tensor = torch.from_numpy(pandas_series.values)   # Convert a panda series to a tensor
print("The data type of new tensor is: ", new_tensor.dtype)
print("The tensor type is: ", new_tensor.type())

# we can use the method tolist() to return a list
torch_to_list = new_tensor.tolist()
print("\nThe list is:", torch_to_list)

########################################### Indexing
print('\n\n--> INDEXING:')

# Consider the following tensor
this_tensor = torch.tensor(list_ints)
# The method item() returns the value of this tensor as a standard Python number. This only works for one element.
print("The first item is", this_tensor[0].item(), "and the first tensor value is", this_tensor[0])
print("The second item is", this_tensor[1].item(), "and the second tensor value is", this_tensor[1])
print("The third item is", this_tensor[2].item(), "and the third tensor value is", this_tensor[2])
print("The last item is", this_tensor[-1].item(), "and the last tensor value is", this_tensor[-1])
# Note that the index_tensor[5] will create an error.

# Now, you'll see how to change the values on certain indexes.
# Suppose you have a tensor as shown here:
# A tensor for showing how to change value according to the index
tensor_sample = torch.tensor([20, 1, 2, 3, 4])
# Assign the value on index 0 as 100:
# Change the value on the index 0 to 100
print("\nInitial value on index 0:", tensor_sample[0])
tensor_sample[0] = 100
print("Modified tensor:", tensor_sample)
# As you can see, the value on index 0 changes. Change the value on index 4 to 0:
# Change the value on the index 4 to 0
print("Initial value on index 4:", tensor_sample[4])
tensor_sample[4] = 0
print("Modified tensor:", tensor_sample)
# The value on index 4 turns to 0.

########################################### Slicing
print('\n\n--> SLICING:')

# If you are familiar with Python, you know that there is a feature called slicing on a list. Tensors support the same
# feature.
# Get the subset of tensor_sample. The subset should contain the values in tensor_sample from index 1 to index 3.
subset_tensor_sample = tensor_sample[1:4]
print("Original tensor sample: ", tensor_sample)
print("The subset of tensor sample:", subset_tensor_sample)
# As a result, the subset_tensor_sample returned only the values on index 1, index 2, and index 3. Then, it stored them
# in a subset_tensor_sample.
# Note: The number on the left side of the colon represents the index of the first value. The number on the right side
# of the colon is always 1 larger than the index of the last value. For example, tensor_sample[1:4] means you get values
# from the index 1 to index 3 (4-1).

# As for assigning values to the certain index, you can also assign the value to the slices:
# Change the value of tensor_sample from index 3 to index 4:
# Change the values on index 3 and index 4
print("\nInital value on index 3 and index 4:", tensor_sample[3:5])
tensor_sample[3:5] = torch.tensor([300.0, 400.0])
print("Modified tensor:", tensor_sample)
# The values on both index 3 and index 4 were changed. The values on other indexes remain the same.

# You can also use a variable to contain the selected indexes and pass that variable to a tensor slice operation as a
# parameter, for example:
# Using variable to contain the selected index, and pass it to slice operation
selected_indexes = [3, 4]
subset_tensor_sample = tensor_sample[selected_indexes]
print("The initial tensor_sample", tensor_sample)
print("The subset of tensor_sample with the values on index 3 and 4: ", subset_tensor_sample)

# You can also assign one value to the selected indexes by using the variable. For example, assign 100,000 to all the
# selected_indexes:
# Using variable to assign the value to the selected indexes
print("The inital tensor_sample", tensor_sample)
selected_indexes = [1, 3]
tensor_sample[selected_indexes] = 100000
print("Modified tensor with one value: ", tensor_sample)
# The values on index 1 and index 3 were changed to 100,000. Others remain the same.
# Note: You can use only one value for the assignment.

########################################### Mean
print('\n\n--> MEAN:')

# Create a tensor with values [1.0, -1, 1, -1]:
# Sample tensor for mathematics calculation methods on tensor
math_tensor = torch.tensor([1.0, -1.0, 1, -1])
print("Tensor example: ", math_tensor)

# Here is the mean method:
mean = math_tensor.mean()  # Calculate the mean for math_tensor
print("The mean of math_tensor: ", mean)

########################################### Standard Deviation
print('\n\n--> STANDARD DEVIATION:')

# The standard deviation can also be calculated by using tensor_obj.std():
standard_deviation = math_tensor.std()  # Calculate the standard deviation for math_tensor
print("The standard deviation of math_tensor: ", standard_deviation)

########################################### Max and Min
print('\n\n--> MAX AND MIN:')

# Now, you'll review another two useful methods: tensor_obj.max() and tensor_obj.min(). These two methods are used for
# finding the maximum value and the minimum value in the tensor.
# Sample for introducing max and min methods
max_min_tensor = torch.tensor([1, 1, 3, 5, 5])
print("Tensor example: ", max_min_tensor)
# Note: There are two minimum numbers as 1 and two maximum numbers as 5 in the tensor. Can you guess how PyTorch is
# going to deal with the duplicates?

# Apply tensor_obj.max() on max_min_tensor:
# Method for finding the maximum value in the tensor
max_val = max_min_tensor.max()
print("Maximum number in the tensor: ", max_val)
# The answer is tensor(5). Therefore, the method tensor_obj.max() is grabbing the maximum value but not the elements
# that contain the maximum value in the tensor.

# Method for finding the minimum value in the tensor
min_val = max_min_tensor.min()
print("Minimum number in the tensor: ", min_val)
# The answer is tensor(1). Therefore, the method tensor_obj.min() is grabbing the minimum value but not the elements
# that contain the minimum value in the tensor.

########################################### Sin
print('\n\n--> SIN:')

# Sin is the trigonometric function of an angle. Again, you will not be introducedvto any mathematic functions. You'll
# focus on Python.
# Create a tensor with 0, π/2 and π. Then, apply the sin function on the tensor. Notice here that the sin() is not a
# method of tensor object but is a function of torch:
pi_tensor = torch.tensor([0, np.pi / 2, np.pi])
sin = torch.sin(pi_tensor)  # Method for calculating the sin result of each element in the tensor
print("The sin result of pi_tensor: ", sin)
# The resultant tensor sin contains the result of the sin function applied to each element in the pi_tensor.
# This is different from the previous methods. For tensor_obj.mean(), tensor_obj.std(), tensor_obj.max(), and
# tensor_obj.min(), the result is a tensor with only one number because these are aggregate methods.
# However, the torch.sin() is not. Therefore, the resultant tensors have the same length as the input tensor.

########################################### Linspace
print('\n\n--> LINSPACE:')

# Create Tensor by torch.linspace()
# A useful function for plotting mathematical functions is torch.linspace() taht returns evenly spaced numbers over a
# specified interval. You specify the starting point of the sequence and the ending point of the sequence. The parameter
# steps indicates the number of samples to generate. Now, you'll work with steps = 5.
len_5_tensor = torch.linspace(-2, 2, steps=5)  # First try on using linspace to create tensor
print("First Try on linspace", len_5_tensor)

# Second try on using linspace to create tensor, now with steps = 9.
len_9_tensor = torch.linspace(-2, 2, steps=9)
print("Second Try on linspace", len_9_tensor)

# Example:
# Construct the tensor within 0 to 360 degree
pi_tensor = torch.linspace(0, 2 * np.pi, 100)
sin_result = torch.sin(pi_tensor)

# Plot the result to get a clearer picture. You must cast the tensor to a numpy array before plotting it.
# Plot sin_result
plt.figure()
plt.plot(pi_tensor.numpy(), sin_result.numpy())
plt.show()

# If you know the trigonometric function, you will notice this is the diagram of the sin result in the range 0 to 360°.

########################################### Tensor Operations
print("\n\nTENSOR OPERATIONS:")

## -- Addition

# You can perform addition between two tensors.
# Note: Tensors must be of the same data type to perform addition as well as other operations.

# Create a tensor u with 1 dimension and 2 elements. Then, create another tensor v with the same number of dimensions
# and the same number of elements:
u = torch.tensor([1, 0])
v = torch.tensor([0, 1])

# tensor + tensor
w = u + v
print("The result of u + v is: ", w)

# Plot the result (u,v,w) to to get a clearer picture.
plotVec([
    {"vector": u.numpy(), "name": 'u', "color": 'r'},
    {"vector": v.numpy(), "name": 'v', "color": 'b'},
    {"vector": w.numpy(), "name": 'w', "color": 'g'}
])
plt.show()

# tensor + scalar
u = torch.tensor([1, 2, 3, -1])
v = u + 1
print("Addition of u with a scalar: ", v)
# The result is simply adding 1 to each element in tensor u.

## -- Multiplication

# Now, you'll review the multiplication between a tensor and a scalar.
# Create a tensor with value [1, 2] and then multiply it by 2:

# tensor * scalar
u = torch.tensor([1, 2])
v = 2 * u
print("The result of 2 * u: ", v)
# The result is tensor([2, 4]), so the code 2 * u multiplies each element in the tensor by 2. This is how you get the
# product between a vector or matrix and a scalar in linear algebra.

# You can use multiplication between two tensors.
# Create two tensors u and v and then multiply them together:

# tensor * tensor
u = torch.tensor([1, 2])
v = torch.tensor([3, 2])
w = u * v
print("The result of u * v", w)

# The result is simply tensor([3, 4]). This result is achieved by multiplying every element in u with the corresponding
# element in the same position v, which is similar to [1 * 3, 2 * 2].

## -- Dot Product

# The dot product is a special operation for a vector that you can use in Torch.
# Here is the dot product of the two tensors u and v:

# Calculate dot product of u, v
u = torch.tensor([1, 2])
v = torch.tensor([3, 2])
print("Dot Product of u, v:", torch.dot(u, v))
# The result is tensor(7). The function is 1 x 3 + 2 x 2 = 7.
