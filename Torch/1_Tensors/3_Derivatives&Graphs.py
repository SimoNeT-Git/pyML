#!/usr/bin/env python
# coding: utf-8

# In this lab, you will learn the basics of differentiation
#  - Derivatives
#  - Partial Derivatives

import torch
import matplotlib.pylab as plt

########################################### Derivatives on single value
print("\n\nDERIVATIVES on single value:")

# Let us create the tensor x and set the parameter requires_grad to true because you are going to take the derivative
# of the tensor.
x = torch.tensor(2.0, requires_grad=True)  # Create a tensor x
print("The tensor x: ", x)

# Then let us create a tensor according to the equation y=x^2.
y = x ** 2
print("The result of y = x^2: ", y)

# Then let us take the derivative with respect to x at x = 2.
# Take the derivative dy/dx = 2x and print out the derivative at the value x = 2.
y.backward()
print("The derivative at x = 2, i.e. dy/dx = 2x: ", x.grad)

# The preceding lines perform the following operation: dy/dx = 2x
print("\nx:")
print('data:', x.data)
print('grad_fn:', x.grad_fn)
print('grad:', x.grad)
print("is_leaf:", x.is_leaf)
print("requires_grad:", x.requires_grad)
print("\ny:")
print('data:', y.data)
print('grad_fn:', y.grad_fn)
print('grad:', y.grad)
print("is_leaf:", y.is_leaf)
print("requires_grad:", y.requires_grad)

# Let us try to calculate the derivative for a more complicated function.
# Calculate the y = x^2 + 2x + 1, then find the derivative
x = torch.tensor(2.0, requires_grad=True)
print("\nThe tensor x: ", x)
y = x ** 2 + 2 * x + 1
print("The result of y = x^2 + 2x + 1: ", y)
y.backward()
print("The derivative at x = 2, i.e. dy/dx = 2x + 2", x.grad)


# We can implement our own custom autograd Functions by subclassing torch.autograd.Function and implementing the
# forward and backward passes which operate on Tensors

class SQ(torch.autograd.Function):

    @staticmethod
    def forward(ctx, i):
        """
        In the forward pass we receive a Tensor containing the input and return
        a Tensor containing the output. ctx is a context object that can be used
        to stash information for backward computation. You can cache arbitrary
        objects for use in the backward pass using the ctx.save_for_backward method.
        """
        result = i ** 2
        ctx.save_for_backward(i)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        """
        In the backward pass we receive a Tensor containing the gradient of the loss
        with respect to the output, and we need to compute the gradient of the loss
        with respect to the input.
        """
        i, = ctx.saved_tensors
        grad_output = 2 * i
        return grad_output


# We can create the object "square function"
sq = SQ.apply

# And apply the object to a variable x, saving the result in a variable y = sq(x), i.e. y = x^2
x = torch.tensor(2.0, requires_grad=True)
print("\nThe tensor x: ", x)
y = sq(x)
print("The result of y = x^2: ", y)
y.backward()
print("The derivative at x = 2, i.e. dy/dx = 2x: ", x.grad)

########################################### Partial Derivatives on single value
print("\n\nPARTIAL DERIVATIVES on single value:")

# We can also calculate Partial Derivatives. Consider the function: f(u,v) = v*u + u^2
# Calculate f(u, v) at u = 1, v = 2
u = torch.tensor(1.0, requires_grad=True)
v = torch.tensor(2.0, requires_grad=True)
print("Variable u is ", u, " and v is ", v)
f = u * v + u ** 2
print("The result of f(u,v) = v*u + u^2: ", f)
# This is equivalent to the following: f(u=1,v=2) = (2)(1)+1^{2} = 3

# Now let us take the derivative with respect to u:
f.backward()
print("The partial derivative with respect to u, i.e. df/du = v + 2u: ", u.grad)
print("The partial derivative with respect to v, i.e. df/dv = u: ", v.grad)

########################################### Derivatives on multiple values
print("\n\nDERIVATIVES on multiple values:")

# Calculate the derivative with respect to a function with multiple values as follows. You use the sum trick to produce
# a scalar valued function and then take the gradient:
x = torch.linspace(-10, 10, 10, requires_grad=True)
y = x ** 2
y_sum = torch.sum(y)

# Take the derivative with respect to multiple value.
y_sum.backward()

# Plot the function and its derivative
plt.figure()
plt.plot(x.detach().numpy(), y.detach().numpy(), label='function')
plt.plot(x.detach().numpy(), x.grad.detach().numpy(), label='derivative')
plt.xlabel('x')
plt.legend()
plt.title('Square function')
plt.show()
# The orange line is the slope of the blue line at the intersection point, which is the derivative of the blue line.
# The method detach() excludes further tracking of operations in the graph, and therefore the subgraph will not record
# operations. This allows us to then convert the tensor to a numpy array.

# The ReLu activation function is an essential function in neural networks. We can take the derivative as follows.
# Take the derivative of ReLu with respect to multiple value. Plot out the function and its derivative
x = torch.linspace(-10, 10, 1000, requires_grad=True)
y = torch.relu(x)
y_sum = y.sum()

# Take the derivative with respect to multiple value.
y_sum.backward()

# Plot the function and its derivative
plt.figure()
plt.plot(x.detach().numpy(), y.detach().numpy(), label='function')
plt.plot(x.detach().numpy(), x.grad.detach().numpy(), label='derivative')
plt.xlabel('x')
plt.legend()
plt.title('ReLu function')
plt.show()
