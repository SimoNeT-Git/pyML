#!/usr/bin/env python
# coding: utf-8

# In this lab, you will learn the basics of tensor operations on 2D tensors.
#  - Types and Shape
#  - Indexing and Slicing
#  - Tensor Operations

import numpy as np
import matplotlib.pyplot as plt
import torch
import pandas as pd

########################################### Types and Shape
print("\n\nTYPES AND SHAPE:")

# The methods and types for 2D tensors is similar to the methods and types for 1D tensors which has been introduced in
# Previous Lab.

# Let us see how to convert a 2D list to a 2D tensor. First, let us create a 3X3 2D tensor. Then let us try to use
# torch.tensor() which we used for converting a 1D list to 1D tensor. Is it going to work?
twoD_list = [[11, 12, 13], [21, 22, 23], [31, 32, 33]]
twoD_tensor = torch.tensor(twoD_list)  # Convert 2D List to 2D Tensor
print("The New 2D Tensor: ", twoD_tensor)

# Bravo! The method torch.tensor() works perfectly. Now, let us try other functions we studied in the Previous Lab.
# Try tensor_obj.ndimension(), tensor_obj.shape, tensor_obj.size()
print("\nThe dimension of twoD_tensor: ", twoD_tensor.ndimension())
print("The shape of twoD_tensor: ", twoD_tensor.shape)
print("The shape of twoD_tensor: ", twoD_tensor.size())
print("The number of elements in twoD_tensor: ", twoD_tensor.numel())
# Because it is a 2D 3x3 tensor, the outputs are correct.

########################################### Numpy to Tensor and viceversa
print("\n\nNUMPY ARRAYS AND TENSORS:")

# Now, let us try converting the tensor to a numpy array and convert the numpy array back to a tensor.
# Convert tensor to numpy array; Convert numpy array to tensor
twoD_numpy = twoD_tensor.numpy()
print("Tensor -> Numpy Array:")
print("The numpy array after converting: ", twoD_numpy)
print("Type after converting: ", twoD_numpy.dtype)

print("================================================")

new_twoD_tensor = torch.from_numpy(twoD_numpy)
print("Numpy Array -> Tensor:")
print("The tensor after converting:", new_twoD_tensor)
print("Type after converting: ", new_twoD_tensor.dtype)
# The result shows the tensor has successfully been converted to a numpy array and then converted back to a tensor.

########################################### From Pandas Series to Tensor
print('\n\n--> FROM PANDAS SERIES TO TENSOR AND LIST:')

# Now let us try to convert a Pandas Dataframe to a tensor. The process is the  Same as the 1D conversion, we can obtain
# the numpy array via the attribute values. Then, we can use torch.from_numpy() to convert the value of the Pandas
# Series to a tensor.
# Try to convert the Panda Dataframe to tensor
df = pd.DataFrame({'a': [11, 21, 31], 'b': [12, 22, 312]})
print("Pandas Dataframe to numpy: ", df.values)
print("Type BEFORE converting: ", df.values.dtype)

print("================================================")

new_tensor = torch.from_numpy(df.values)
print("Tensor AFTER converting: ", new_tensor)
print("Type AFTER converting: ", new_tensor.dtype)

########################################### Indexing and Slicing
print('\n\n--> INDEXING AND SLICING:')

# You can use rectangular brackets to access the different elements of the tensor.
# You can access the 2nd-row 3rd-column: you simply use the square brackets and the indices corresponding to the element
# that you want.

# Use tensor_obj[row, column] and tensor_obj[row][column] to access certain position
tensor_example = torch.tensor([[11, 12, 13], [21, 22, 23], [31, 32, 33]])
print("What is the value on 2nd-row 3rd-column? ", tensor_example[1, 2])
print("What is the value on 2nd-row 3rd-column? ", tensor_example[1][2])

# Use the method above, we can access the 1st-row 1st-column by >tensor_example[0][0]
print("First element of first row is", tensor_example[0][0])

# But what if we want to get the value on both 1st-row 1st-column and 1st-row 2nd-column?

# Use tensor_obj[begin_row_number: end_row_number, begin_column_number: end_column number] 
# and tensor_obj[row][begin_column_number: end_column number] to do the slicing
tensor_example = torch.tensor([[11, 12, 13], [21, 22, 23], [31, 32, 33]])
print("What is the value on 1st-row first two columns? ", tensor_example[0, 0:2])
print("What is the value on 1st-row first two columns? ", tensor_example[0][0:2])

# <!--Empty Space for separating topics-->

# But we can't combine using slicing on row and pick one column by using the code
# tensor_obj[begin_row_number: end_row_number][begin_column_number: end_column number].
# The reason is that the slicing will be applied on the tensor first. The result type will be a two dimension again.
# The second bracket will no longer represent the index of the column it will be the index of the row at that time.
# Let us see an example.
# Give an idea on tensor_obj[number: number][number]
tensor_example = torch.tensor([[11, 12, 13], [21, 22, 23], [31, 32, 33]])
sliced_tensor_example = tensor_example[1:3]
print("1. Slicing step on tensor_example: ")
print("Result after tensor_example[1:3]: ", sliced_tensor_example)
print("Dimension after tensor_example[1:3]: ", sliced_tensor_example.ndimension())
print("================================================")
print("2. Pick an index on sliced_tensor_example: ")
print("Result after sliced_tensor_example[1]: ", sliced_tensor_example[1])
print("Dimension after sliced_tensor_example[1]: ", sliced_tensor_example[1].ndimension())
print("================================================")
print("3. Combine these step together:")
print("Result: ", tensor_example[1:3][1])
print("Dimension: ", tensor_example[1:3][1].ndimension())
# See the results and dimensions in 2 and 3 are the same. Both of them contains the 3rd row in the tensor_example, but
# not the last two values in the 3rd column.

# So how can we get the elements in the 3rd column with the last two rows?
# Let's see the code below.
# Use tensor_obj[begin_row_number: end_row_number, begin_column_number: end_column number]
tensor_example = torch.tensor([[11, 12, 13], [21, 22, 23], [31, 32, 33]])
print("What is the value on 3rd-column last two rows? ", tensor_example[1:3, 2])
# Fortunately, the code tensor_obj[begin_row_number: end_row_number, begin_column_number: end_column number]
# it still works.

########################################### Tensor Operations
print("\n\nTENSOR OPERATIONS:")

# We can also do some calculations on 2D tensors.

# -- Tensor Addition

# Calculate [[1, 0], [0, 1]] + [[2, 1], [1, 2]]
X = torch.tensor([[1, 0], [0, 1]])
Y = torch.tensor([[2, 1], [1, 2]])
X_plus_Y = X + Y
print("The result of X + Y: ", X_plus_Y)

# -- Scalar Multiplication

# Multiplying a tensor by a scalar is identical to multiplying a matrix by a scaler. If you multiply the matrix Y by
# the scalar 2, you simply multiply every element in the matrix by 2.

# Calculate 2 * [[2, 1], [1, 2]]
Y = torch.tensor([[2, 1], [1, 2]])
two_Y = 2 * Y
print("The result of 2Y: ", two_Y)

# -- Element-wise Product/Hadamard Product

# Multiplication of two tensors corresponds to an element-wise product or Hadamard product.
# Consider matrix the X and Y with the same size. The Hadamard product corresponds to multiplying each of
# the elements at the same position, that is, multiplying elements with the same color together. The result is a new
# matrix that is the same size as matrix X and Y:

# The code below calculates the element-wise product of the tensor <strong>X</strong> and <strong>Y</strong>:
# Calculate [[1, 0], [0, 1]] * [[2, 1], [1, 2]]
X = torch.tensor([[1, 0], [0, 1]])
Y = torch.tensor([[2, 1], [1, 2]])
X_times_Y = X * Y
print("The result of X * Y: ", X_times_Y)

# -- Matrix Multiplication

# We can also apply matrix multiplication to two tensors, if you have learned linear algebra, you should know that in
# the multiplication of two matrices order matters. This means if X * Y is valid, it does not mean Y * X is valid.
# The number of columns of the matrix on the left side of the multiplication sign must equal to the number of rows of
# the matrix on the right side.

# We use torch.mm() for calculating the multiplication between tensors with different sizes.
# Calculate [[0, 1, 1], [1, 0, 1]] * [[1, 1], [1, 1], [-1, 1]]
A = torch.tensor([[0, 1, 1], [1, 0, 1]])
B = torch.tensor([[1, 1], [1, 1], [-1, 1]])
A_times_B = torch.mm(A, B)
print("The result of A * B: ", A_times_B)
