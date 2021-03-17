#!/usr/bin/env python
# coding: utf-8

# Binary classification and model selection in synthetic data

import matplotlib.pyplot as plt
import numpy as np
from sklearn.kernel_ridge import KernelRidge
from sklearn import model_selection
from tabulate import tabulate
from utils import create_random_data, data_split, plot_dataset, plot_separation


def binary_classif_error(y_true, y_pred):
    return np.mean(np.sign(y_pred) != y_true)


# ### ---> Part 1: Warmup
#
# You will use a regularized least square model for classification on a toy dataset. You will explore the differences
# between training and test error, and how the regularization parameter affects them.
#
# We will use cross-validation to estimate the best values for the parameter.
#
# Data Generation
#
# The create_random_data function is used throughout this lab to generate random datasets. If you want to see how it
# works, look at the lab1_utils.py file in this folder.
X, y = create_random_data(n_samples=100, noise_level=1.3, dataset="linear")
print("%d samples, %d features" % X.shape)
plot_dataset(X, y)
plt.show()

# Splitting the data into train and test
#
# We use another function defined in the lab1_utils.py file data_split to subdivide the data into 80 training samples,
# and (correspondingly) 20 test samples.
X_train, X_test, y_train, y_test = data_split(X, y, n_train=80)
print("Generated %d training samples, %d test samples" % (X_train.shape[0], X_test.shape[0]))

# Training a linear ridge-regression model
#
# We will use the sklearn.kernel_ridge.KernelRidge class from scikit-learn to define our models, specifying that we want
# the "linear" kernel. Then the only parameter is the regularization parameter which we look into in this section.
regularization = 0.1
model = KernelRidge(regularization, kernel="linear")
model.fit(X_train, y_train)

train_preds = model.predict(X_train)
test_preds = model.predict(X_test)

print("Training error: %.2f%%" % (binary_classif_error(y_train, train_preds) * 100))
print("Test error: %.2f%%" % (binary_classif_error(y_test, test_preds) * 100))

plot_separation(X_train, y_train, model)
plt.show()

# Exploring the effect of different parameters¶
#
# First we change the regularization parameter, observing what happens. Since the data is very low-dimensional, th
# change is not visible until reaching very large amounts of regularization.
#
# For now calculate the test errors (no cross-validation is needed).

# 1. Change the regularization parameter
reg_values = np.geomspace(1e-4, 5e3, num=50)
test_errors = []
for reg in reg_values:
    model = KernelRidge(reg, kernel="linear")
    model.fit(X_train, y_train)
    test_preds = model.predict(X_test)
    test_errors.append(binary_classif_error(y_test, test_preds))

fig, ax = plt.subplots()
ax.semilogx(reg_values, test_errors)
ax.set_xlabel("Regularization")
ax.set_ylabel("Test error")
plt.show()

# 2. Change in number of data-points
num_points = np.arange(5, 1000, 10)
np_test_errors = []
model = KernelRidge(1e-4, kernel="linear")
for points in num_points:
    X, y = create_random_data(points + 20, 1, seed=932)
    X_train, X_test, y_train, y_test = data_split(X, y, n_train=points)
    model.fit(X_train, y_train)
    test_preds = model.predict(X_test)
    np_test_errors.append(binary_classif_error(y_test, test_preds))

fig, ax = plt.subplots()
ax.plot(num_points, np_test_errors)
ax.set_xlabel("Number of training points")
ax.set_ylabel("Test error")
plt.show()

# 3. Amount of noise in the data
data_noise = [0.3, 0.5, 1.0, 2.0]
noise_test_errors = []
model = KernelRidge(1e-4, kernel="linear")
for noise in data_noise:
    X, y = create_random_data(200, noise, seed=932)
    X_train, X_test, y_train, y_test = data_split(X, y, n_train=int(200 * 0.8))
    model.fit(X_train, y_train)
    test_preds = model.predict(X_test)
    noise_test_errors.append(binary_classif_error(y_test, test_preds))

fig, ax = plt.subplots()
ax.plot(data_noise, noise_test_errors)
ax.set_xlabel("Data noise")
ax.set_ylabel("Test error")
plt.show()

# Cross-Validation
#
# Find the optimal value for the regularization parameter. By using K-fold cross-validation, you can increase the
# confidence that good parameter settings will still be valid on the test set.
#
# Remember that you should only look at the test set at the very end, to avoid overfitting to it. In a real-world
# setting, you will not know what the test data looks like, and relying on cross-validation is one way to reduce
# overfitting to the training data.
#
# For this exercise it is easy to check different values of the regularization parameter by hand. For more complex
# scenarios, scikit-learn includes some useful classes which greatly reduce the amount of boilerplate code needed for
# tuning hyperparameters. For example, look at sklearn.model_selection.GridSearchCV and
# sklearn.model_selection.RandomizedSearchCV
#
# In the case of our data, we have already seen that several regularization parameters seem to work well.
# In the following it is sufficient to find one of them.
reg_params = [1e-7, 1e-5, 1e-3, 1e-1, 1e1, 1e3]
kfold_cv = model_selection.KFold(n_splits=5, shuffle=True, random_state=102)
errors = {rp: [] for rp in reg_params}

# Loop through all possible regularization parameters
for rp in reg_params:
    model = KernelRidge(rp, kernel="linear")
    # Run K-Fold CV (on train data)
    for train_index, val_index in kfold_cv.split(X_train):
        model.fit(X_train[train_index], y_train[train_index])
        test_preds = model.predict(X_train[val_index])
        error = binary_classif_error(y_train[val_index], test_preds)
        errors[rp].append(error)

# Calculate the best error and corresponding regularization parameter
min_rp, min_es = min(errors.items(), key=lambda kv: np.mean(kv[1]))
print("The regularization parameter with minimal error is %e" % min_rp)
print("Achieving a 5-fold CV average error of %.2f%%" % (np.mean(min_es) * 100))

# ### ---> Part 2: Kernel ridge regression
#
# Here we will use the same model as in Part 1, but instead of taking the linear kernel, which is equivalent to
# performing linear ridge regression, we take a different kernel.
#
# Different kernels can have different parameters. For example, the Gaussian (or RBF) kernel is defined by its
# length-scale, or sigma.
#
# To use the Gaussian kernel with the KernelRidge estimator, pass kernel="rbf" to it instead of kernel="linear". Note
# that in scikit-learn, the Gaussian kernel has a gamma parameter which is defined as $\gamma = \dfrac{1}{2\sigma^2}$.
# So be careful that a large $\gamma$ corresponds to small $\sigma$ and viceversa.
# Tasks:
#
#    1) Perform parameter tuning for kernel ridge regression with a Gaussian kernel:
#        - Try different (gamma, regularization) pairs and compare the obtained training and test errors
#        - Fix the regularization and observe the effect of changing the length-scale gamma
#        - Fix gamma and observe the effect of changing the regularization
#        - Do you notice (and if so, when) any overfitting/oversmoothing effects? Try to confirm your results by
#          exploring a range of parameters and plotting the training and test errors.
#
#    2) Consider the Polynomial kernel now (can be selected with kernel="polynomial") and perform parameter tuning over
#    its parameters. Note that the polynomial kernel has three different parameters (gamma, degree, and coef0). Compare
#    the performances of the polynomial and Gaussian kernels on the circles and moons datasets with respect to the
#    training set size (e.g. [10, 20, 50, 100, 1000]) and the amount of regularization
#    (e.g. [0.5, 0.1, 0.01, 0.001, 0.0001]).

# Generate circles data
N = 1000
X, y = create_random_data(n_samples=N, noise_level=0.05, dataset="circles", seed=932)
plot_dataset(X, y)
plt.show()
X_train, X_test, y_train, y_test = data_split(X, y, n_train=int(N * 0.8))

# Set-up KRR with Gaussian kernel example
g_model = KernelRidge(0.01, kernel="rbf", gamma=0.01)
g_model.fit(X_train, y_train)
g_err = binary_classif_error(y_test, g_model.predict(X_test))
print("Test error of Gaussian kernel with gamma=%.2f, regularization=%.2f : %.2f%%" %
      (g_model.gamma, g_model.alpha, g_err * 100))

plot_separation(X_test, y_test, g_model)
plt.show()

# Find the best parameter values
# Iterate through different regularization parameter values
reg_values = np.geomspace(1e-4, 5e3, num=50)
test_errors, train_errors = [], []
for reg in reg_values:
    g_model = KernelRidge(reg, kernel="rbf", gamma=0.01)
    g_model.fit(X_train, y_train)
    test_errors.append(binary_classif_error(y_test, g_model.predict(X_test)))
    train_errors.append(binary_classif_error(y_train, g_model.predict(X_train)))

fig, ax = plt.subplots()
ax.semilogx(reg_values, test_errors, label='test')
ax.semilogx(reg_values, train_errors, label='train')
plt.legend()
ax.set_xlabel("Regularization Parameter")
ax.set_ylabel("Error")
plt.show()

# Iterate through different gamma values
gamma_values = np.geomspace(1e-4, 5e3, num=50)
test_gerrors, train_gerrors = [], []
for g in gamma_values:
    g_model = KernelRidge(1e-4, kernel="rbf", gamma=g)
    g_model.fit(X_train, y_train)
    test_gerrors.append(binary_classif_error(y_test, g_model.predict(X_test)))
    train_gerrors.append(binary_classif_error(y_train, g_model.predict(X_train)))

fig, ax = plt.subplots()
ax.semilogx(gamma_values, test_gerrors, label='test')
ax.semilogx(gamma_values, train_gerrors, label='train')
plt.legend()
ax.set_xlabel("Gamma Value")
ax.set_ylabel("Error")
plt.show()
# # Does the Gaussian kernel overfit? If so, for which parameters?

# Compare Gaussian and Polynomial kernels on the circles and moons datasets
#
# Since the Polynomial kernel has many parameters, you can perform a full grid search to understand how these parameters
# interact. We have provided you with a skeleton code for the grid search.
param_grid_poly = {"coef0": [0, 1], "degree": [2, 3, 4], "gamma": [0.01, 1, 10]}
param_grid_gauss = {"gamma": [0.01, 1, 10]}
reg_values = [0.0001, 0.001, 0.01, 0.1, 0.5]
train_size = [10, 20, 50, 100, 1000]

# ## Circle dataset
#
# Generate such data
N = 1000
X, y = create_random_data(n_samples=N, noise_level=0.05, dataset="circles", seed=932)
X_train, X_test, y_train, y_test = data_split(X, y, n_train=int(N * 0.8))

# Set-up KRR with Gaussian kernel example
params_poly, error_poly, error_train_poly = [], [], []
params_gauss, error_gauss, error_train_gauss = [], [], []
for reg in reg_values:
    # Polynomial kernel
    model_poly = KernelRidge(reg, kernel="polynomial")
    best_model_poly = model_selection.GridSearchCV(model_poly, param_grid_poly)  # , scoring="accuracy")
    best_model_poly.fit(X_train, y_train)
    error_poly.append(binary_classif_error(y_test, best_model_poly.best_estimator_.predict(X_test)))
    error_train_poly.append(binary_classif_error(y_train, best_model_poly.best_estimator_.predict(X_train)))
    best_estim_poly = best_model_poly.best_estimator_
    params_poly.append([reg, best_estim_poly.coef0, best_estim_poly.degree,
                        best_estim_poly.gamma, best_model_poly.best_score_])

    # Gaussian kernel
    model_gauss = KernelRidge(reg, kernel="rbf")
    best_model_gauss = model_selection.GridSearchCV(model_gauss, param_grid_gauss)  # , scoring="accuracy")
    best_model_gauss.fit(X_train, y_train)
    error_gauss.append(binary_classif_error(y_test, best_model_gauss.best_estimator_.predict(X_test)))
    error_train_gauss.append(binary_classif_error(y_train, best_model_gauss.best_estimator_.predict(X_train)))
    best_estim_gauss = best_model_gauss.best_estimator_
    params_gauss.append([reg, best_estim_gauss.gamma, best_model_gauss.best_score_])

print('Polynomial Kernel')
print(tabulate(params_poly, headers=['reg value', 'coef0', 'degree', 'gamma', 'CV score'], tablefmt="fancy_grid"))
print('Gaussian Kernel')
print(tabulate(params_gauss, headers=['reg value', 'gamma', 'CV score'], tablefmt="fancy_grid"))

scores_poly = [params_poly[k][4] for k in range(len(params_poly))]
scores_gauss = [params_gauss[k][2] for k in range(len(params_gauss))]

# plt.figure()
# plt.plot(reg_values, scores_poly, label='polynomial')
# plt.plot(reg_values, scores_gauss, label='gaussian')
# plt.legend()
# plt.xlabel('Regularization parameters')
# plt.ylabel('CV score')
# plt.title('Circle Dataset')
# plt.show()

plt.figure()
plt.plot(reg_values, error_poly, 'k-', label='polynomial test')
plt.plot(reg_values, error_train_poly, 'k--', label='polynomial train')
plt.plot(reg_values, error_gauss, 'r-', label='gaussian test')
plt.plot(reg_values, error_train_gauss, 'r--', label='gaussian train')
plt.legend()
plt.xlabel('Regularization parameters')
plt.ylabel('Error')
plt.title('Circle Dataset')
plt.show()

# model = KernelRidge(1.0, kernel="polynomial")
# gs = model_selection.GridSearchCV(model, param_grid)  # , scoring="accuracy")
# # Fit the Grid Search
# gs.fit(X_train, y_train)
# # Have a look at the results (hint: look at the cv_results_ attribute)
# param_results = []
# for i in range(len(gs.cv_results_.get('params'))):
#     parresi = list(gs.cv_results_.get('params')[i].values())
#     parresi.append(gs.cv_results_.get('mean_test_score')[i] * 100)
#     param_results.append(parresi)
# print(tabulate(param_results, headers=['coef0', 'degree', 'gamma', 'test score'], tablefmt="fancy_grid"))
#
# # Evaluate results
# print("Best estimator: ", gs.best_estimator_)
# gs.best_estimator_.fit(X_train, y_train)
# test_preds = gs.best_estimator_.predict(X_test)
# print("Test error: %.2f" % (binary_classif_error(y_test, test_preds)))

model_poly = KernelRidge(0.0001, kernel="polynomial")
best_model_poly = model_selection.GridSearchCV(model_poly, param_grid_poly)  # , scoring="accuracy")
model_gauss = KernelRidge(0.0001, kernel="rbf")
best_model_gauss = model_selection.GridSearchCV(model_gauss, param_grid_gauss)  # , scoring="accuracy")
scores_poly, error_poly, error_train_poly = [], [], []
scores_gauss, error_gauss, error_train_gauss = [], [], []
for tr_size in train_size:
    # Generate circles data
    X, y = create_random_data(n_samples=tr_size, noise_level=0.05, dataset="circles", seed=932)
    X_train, X_test, y_train, y_test = data_split(X, y, n_train=int(tr_size * 0.8))

    # Polynomial kernel
    best_model_poly.fit(X_train, y_train)
    error_poly.append(binary_classif_error(y_test, best_model_poly.best_estimator_.predict(X_test)))
    error_train_poly.append(binary_classif_error(y_train, best_model_poly.best_estimator_.predict(X_train)))
    scores_poly.append(best_model_poly.best_score_)

    # Gaussian kernel
    best_model_gauss.fit(X_train, y_train)
    error_gauss.append(binary_classif_error(y_test, best_model_gauss.best_estimator_.predict(X_test)))
    error_train_gauss.append(binary_classif_error(y_train, best_model_poly.best_estimator_.predict(X_train)))
    scores_gauss.append(best_model_gauss.best_score_)

# plt.figure()
# plt.plot(train_size, scores_poly, label='polynomial')
# plt.plot(train_size, scores_gauss, label='gaussian')
# plt.legend()
# plt.xlabel('Training Size')
# plt.ylabel('CV score')
# plt.title('Circle Dataset')
# plt.show()

plt.figure()
plt.plot(train_size, error_poly, 'k-', label='polynomial test')
plt.plot(train_size, error_train_poly, 'k--', label='polynomial train')
plt.plot(train_size, error_gauss, 'r-', label='gaussian test')
plt.plot(train_size, error_train_gauss, 'r--', label='gaussian train')
plt.legend()
plt.xlabel('Training size')
plt.ylabel('Error')
plt.title('Circle Dataset')
plt.show()

# ## Moons dataset
#
# Generate such data
X, y = create_random_data(n_samples=N, noise_level=0.05, dataset="moons", seed=932)
plot_dataset(X, y)
X_train, X_test, y_train, y_test = data_split(X, y, n_train=int(N * 0.8))

# Set-up KRR with Gaussian kernel example, then compare Gaussian and Polynomial kernels on the circle and moon datasets
#
# Since the Polynomial kernel has many parameters, you can perform a full grid search to understand how these parameters
# interact. We have provided you with a skeleton code for the grid search.
params_poly, error_poly, error_train_poly = [], [], []
params_gauss, error_gauss, error_train_gauss = [], [], []
for reg in reg_values:
    # Polynomial kernel
    model_poly = KernelRidge(reg, kernel="polynomial")
    best_model_poly = model_selection.GridSearchCV(model_poly, param_grid_poly)  # , scoring="accuracy")
    best_model_poly.fit(X_train, y_train)
    error_poly.append(binary_classif_error(y_test, best_model_poly.best_estimator_.predict(X_test)))
    error_train_poly.append(binary_classif_error(y_train, best_model_poly.best_estimator_.predict(X_train)))
    best_estim_poly = best_model_poly.best_estimator_
    params_poly.append([reg, best_estim_poly.coef0, best_estim_poly.degree,
                        best_estim_poly.gamma, best_model_poly.best_score_])

    # Gaussian kernel
    model_gauss = KernelRidge(reg, kernel="rbf")
    best_model_gauss = model_selection.GridSearchCV(model_gauss, param_grid_gauss)  # , scoring="accuracy")
    best_model_gauss.fit(X_train, y_train)
    error_gauss.append(binary_classif_error(y_test, best_model_gauss.best_estimator_.predict(X_test)))
    error_train_gauss.append(binary_classif_error(y_train, best_model_gauss.best_estimator_.predict(X_train)))
    best_estim_gauss = best_model_gauss.best_estimator_
    params_gauss.append([reg, best_estim_gauss.gamma, best_model_gauss.best_score_])

print('Polynomial Kernel')
print(tabulate(params_poly, headers=['reg value', 'coef0', 'degree', 'gamma', 'CV score'], tablefmt="fancy_grid"))
print('Gaussian Kernel')
print(tabulate(params_gauss, headers=['reg value', 'gamma', 'CV score'], tablefmt="fancy_grid"))

scores_poly = [params_poly[k][4] for k in range(len(params_poly))]
scores_gauss = [params_gauss[k][2] for k in range(len(params_gauss))]

# plt.figure()
# plt.plot(reg_values, scores_poly, label='polynomial')
# plt.plot(reg_values, scores_gauss, label='gaussian')
# plt.legend()
# plt.xlabel('Regularization parameters')
# plt.ylabel('CV score')
# plt.title('Moon Dataset')
# plt.show()

plt.figure()
plt.plot(reg_values, error_poly, 'k-', label='polynomial test')
plt.plot(reg_values, error_train_poly, 'k--', label='polynomial train')
plt.plot(reg_values, error_gauss, 'r-', label='gaussian test')
plt.plot(reg_values, error_train_gauss, 'r--', label='gaussian train')
plt.legend()
plt.xlabel('Regularization parameters')
plt.ylabel('Error')
plt.title('Moon Dataset')
plt.show()

model_poly = KernelRidge(0.0001, kernel="polynomial")
best_model_poly = model_selection.GridSearchCV(model_poly, param_grid_poly)  # , scoring="accuracy")
model_gauss = KernelRidge(0.0001, kernel="rbf")
best_model_gauss = model_selection.GridSearchCV(model_gauss, param_grid_gauss)  # , scoring="accuracy")
scores_poly, error_poly, error_train_poly = [], [], []
scores_gauss, error_gauss, error_train_gauss = [], [], []
for tr_size in train_size:
    # generate moons data
    X, y = create_random_data(n_samples=tr_size, noise_level=0.05, dataset="moons", seed=932)
    X_train, X_test, y_train, y_test = data_split(X, y, n_train=int(tr_size * 0.8))

    # Polynomial kernel
    best_model_poly.fit(X_train, y_train)
    error_poly.append(binary_classif_error(y_test, best_model_poly.best_estimator_.predict(X_test)))
    error_train_poly.append(binary_classif_error(y_train, best_model_poly.best_estimator_.predict(X_train)))
    scores_poly.append(best_model_poly.best_score_)

    # Gaussian kernel
    best_model_gauss.fit(X_train, y_train)
    error_gauss.append(binary_classif_error(y_test, best_model_gauss.best_estimator_.predict(X_test)))
    error_train_gauss.append(binary_classif_error(y_train, best_model_gauss.best_estimator_.predict(X_train)))
    scores_gauss.append(best_model_gauss.best_score_)

# plt.figure()
# plt.plot(train_size, scores_poly, label='polynomial')
# plt.plot(train_size, scores_gauss, label='gaussian')
# plt.legend()
# plt.xlabel('Training Size')
# plt.ylabel('CV score')
# plt.title('Moon Dataset')
# plt.show()

plt.figure()
plt.plot(train_size, error_poly, 'k-', label='polynomial test')
plt.plot(train_size, error_train_poly, 'k--', label='polynomial train')
plt.plot(train_size, error_gauss, 'r-', label='gaussian test')
plt.plot(train_size, error_train_gauss, 'r--', label='gaussian train')
plt.legend()
plt.xlabel('Training size')
plt.ylabel('Error')
plt.title('Moon Dataset')
plt.show()


# ### ---> Part 3: Challenge
#
# The challenge consists in a learning task using a real dataset, namely USPS (United States Postal Service): This
# dataset contains a number of handwritten digits images. The problem is to train the best KR classifier that is able to
# discriminate between the digits 1 and 7. The data should be in the data folder, please see the load_data() function.

def load_USPS_data():
    from scipy.io import loadmat
    """Loads the USPS one and seven digits."""
    one = loadmat("./Data/one_train.mat")["one_train"]
    seven = loadmat("./Data/seven_train.mat")["seven_train"]
    X = np.concatenate((one, seven), 0)
    Y = np.ones((X.shape[0],))
    Y[one.shape[0]:] = -1
    return X, Y


# We provide an example below of how to load the data, run a simple linear classifier, and save the results to a file.
X, y = load_USPS_data()
print('Shape is: ', X.shape, y.shape)
X_train, X_test, y_train, y_test = data_split(X, y, n_train=int(len(X) * 0.75))

# Basic model settings
regularization = 200

# Try different gammas
train_errors = []
test_errors = []
gammas = np.linspace(1e-7, 1, 100)
k = KernelRidge(regularization, kernel="polynomial")
for gam in gammas:
    k.gamma = gam
    k.fit(X_train, y_train)
    train_err = binary_classif_error(y_train, k.predict(X_train))
    test_err = binary_classif_error(y_test, k.predict(X_test))
    train_errors.append(train_err)
    test_errors.append(test_err)

plt.figure()
plt.plot(gammas, test_errors, label="Test")
plt.plot(gammas, train_errors, label="Train")
plt.legend()
plt.xlabel("Gamma Value")
plt.ylabel("Error")

gamma_choice = np.mean(gammas[np.where(np.asarray(test_errors) == min(test_errors))])

# Try different degrees
train_errors = []
test_errors = []
degrees = np.arange(0, 6)
k = KernelRidge(regularization, kernel="polynomial", gamma=gamma_choice)
for deg in degrees:
    k.degree = deg
    k.fit(X_train, y_train)
    train_err = binary_classif_error(y_train, k.predict(X_train))
    test_err = binary_classif_error(y_test, k.predict(X_test))
    train_errors.append(train_err)
    test_errors.append(test_err)

plt.figure()
plt.plot(degrees, train_errors, label="Train")
plt.plot(degrees, test_errors, label="Test")
plt.legend()
plt.xlabel("Degree Value")
plt.ylabel("Error")

degree_choice = int(np.mean(degrees[np.where(np.asarray(test_errors) == min(test_errors))]))

# Try different coef0
train_errors = []
test_errors = []
coefs = np.linspace(1e-3, 1, 100)
k = KernelRidge(regularization, kernel="polynomial", gamma=gamma_choice, degree=degree_choice)
for coef in coefs:
    k.coef0 = coef
    k.fit(X_train, y_train)
    train_err = binary_classif_error(y_train, k.predict(X_train))
    test_err = binary_classif_error(y_test, k.predict(X_test))
    train_errors.append(train_err)
    test_errors.append(test_err)

plt.figure()
plt.plot(coefs, train_errors, label="Train")
plt.plot(coefs, test_errors, label="Test")
plt.legend()
plt.xlabel("Coef0 Value")
plt.ylabel("Error")
plt.show()

coef0_choice = np.mean(coefs[np.where(np.asarray(test_errors) == min(test_errors))])

# Final choice
finalk = KernelRidge(alpha=200, kernel="polynomial", gamma=gamma_choice, degree=degree_choice, coef0=coef0_choice)
finalk.fit(X_train, y_train)
finalerr = binary_classif_error(y_test, k.predict(X_test))
print('Final error:', finalerr)
