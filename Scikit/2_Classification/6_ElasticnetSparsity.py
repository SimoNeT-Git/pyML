#!/usr/bin/env python
# coding: utf-8

# Sparsity

import numpy as np
from sklearn.linear_model import ElasticNet, ElasticNetCV
import matplotlib.pyplot as plt
from utils import train_test_data

# ### ---> Part 1: Warmup

# ## Noiseless data
#
# First, generate data without noise: y = X * beta_star, where beta_star has n_informative_features non-zero
# entries and X has shape (n_samples, n_features).
#
# We split it into testing and training parts.
n_samples = 100
n_features = 200
n_informative_features = 50

# first, use noiseless data, split in train and test/validation parts
X_train, X_test, y_train, y_test = train_test_data(n_samples, n_features, n_informative_features,
                                                   noise_level=0.)
print("Training dataset shape:", X_train.shape)
print("Testing dataset shape:", X_test.shape)

# In sklearn, the objective function of the ElasticNet optimization is:
# 1/(2*n_samples)*Norm^2{y - X*beta}_2  + alpha*(l1_ratio*Norm{beta}_1 + (1-l1_ratio)/2*Norm^2{beta}_2)
#
# For more information, read the docstring after displaying it in the next cell (you can close the documentation popup
# afterwards by clicking on the cross or hitting Esc).
# Instanciate a classifier with arbitrary values for L1 and L2 penalization
clf = ElasticNet(alpha=0.01, l1_ratio=0.5)

# Fit the model and print its first coefficients. Beware that sklearn fits an intercept by default.
clf.fit(X_train, y_train)
print("50 first coefficients of estimated w:\n", clf.coef_[:50])
print("Intercept: %f" % clf.intercept_)
print("Non-zero coefficients: %d out of %d" % ((clf.coef_ != 0.).sum(), clf.coef_.shape[0]))
print("Training error: %.4f percent" % (np.mean((y_train - clf.predict(X_train)) ** 2) * 100))
print("Testing error: %.4f percent" % (np.mean((y_test - clf.predict(X_test)) ** 2) * 100))
# Note: that y_pred_train = clf.predict(X_train) = X_train @ clf.coef_ + clf.intercept_, where @ is the dot product

# For a fixed alpha, test the influence of l1_ratio on the sparsity of the solution and on the behaviors of the train
# and test errors. For ``l1_ratio = 0`` the penalty is an L2 penalty. ``For l1_ratio = 1`` it is an L1 penalty.
# For ``0 < l1_ratio < 1``, the penalty is a combination of L1 and L2.
alpha_val = 0.01
l1_ratios = np.geomspace(1e-4, 1e0, num=100)

train_errs = np.zeros(len(l1_ratios))
test_errs = np.zeros_like(train_errs)
sparsity = np.zeros_like(train_errs)
for i, l1_ratio in enumerate(l1_ratios):
    clf = ElasticNet(alpha=alpha_val, l1_ratio=l1_ratio)
    clf.fit(X_train, y_train)
    train_errs[i] = np.mean((y_train - clf.predict(X_train)) ** 2)
    test_errs[i] = np.mean((y_test - clf.predict(X_test)) ** 2)
    sparsity[i] = (clf.coef_ != 0.).sum()  # number of non-zero elements in clf.coef_

fig, ax = plt.subplots()
ax.semilogx(l1_ratios, test_errs, label='Test')
ax.semilogx(l1_ratios, train_errs, label='Train')
ax.set_xlabel("l1_ratio")
ax.set_ylabel("Error")
ax.legend()
plt.title('Noiseless Data')
plt.show()

fig, ax = plt.subplots()
ax.semilogx(l1_ratios, sparsity)
ax.set_xlabel('l1_ratio')
ax.set_ylabel(r'$||w||_0$')
plt.title('Noiseless Data')
plt.show()

# What happens when alpha becomes too big? ``alpha = 0`` is equivalent to an ordinary least square, solved by the
# class `LinearRegression` object.
l1_val = 0.3
alphas = np.geomspace(1e-6, 1e3, num=100)

train_errs = np.zeros(len(alphas))
test_errs = np.zeros_like(train_errs)
sparsity = np.zeros_like(train_errs)
for i, alpha in enumerate(alphas):
    clf = ElasticNet(alpha=alpha, l1_ratio=l1_val, max_iter=4000)
    clf.fit(X_train, y_train)
    train_errs[i] = np.mean((y_train - clf.predict(X_train)) ** 2)
    test_errs[i] = np.mean((y_test - clf.predict(X_test)) ** 2)
    sparsity[i] = (clf.coef_ != 0.).sum()  # number of non-zero elements in clf.coef_

fig, ax = plt.subplots()
ax.semilogx(alphas, test_errs, label='Test')
ax.semilogx(alphas, train_errs, label='Train')
ax.set_xlabel("alpha")
ax.set_ylabel("Error")
ax.legend()
plt.title('Noiseless Data')
plt.show()

fig, ax = plt.subplots()
ax.semilogx(alphas, sparsity)
ax.set_xlabel('alpha')
ax.set_ylabel(r'$||w||_0$')
plt.title('Noiseless Data')
plt.show()

# ## Noisy data
#
# Do the same analysis as above, this time when the observations y are corrupted by additive Gaussian noise:
# y = X * beta_star + epsilon
noise_level = 0.5
X_train, X_test, y_train, y_test = train_test_data(n_samples, n_features, n_informative_features,
                                                   noise_level=noise_level)

clf = ElasticNet(alpha=0.01, l1_ratio=0.4)
clf.fit(X_train, y_train)
print("50 first coefficients of estimated w:\n", clf.coef_[:50])
print("Intercept: %f" % clf.intercept_)
print("Non-zero coefficients: %d out of %d" % ((clf.coef_ != 0.).sum(), clf.coef_.shape[0]))
print("Training error: %.4f percent" % (np.mean((y_train - clf.predict(X_train)) ** 2) * 100))
print("Testing error: %.4f percent" % (np.mean((y_test - clf.predict(X_test)) ** 2) * 100))

# For a fixed alpha, test the influence of l1_ratio on the sparsity of the solution and on the behaviors of the train
# and test errors:
train_errs = np.zeros(len(l1_ratios))
test_errs = np.zeros_like(train_errs)
sparsity = np.zeros_like(train_errs)
for i, l1_ratio in enumerate(l1_ratios):
    clf = ElasticNet(alpha=alpha_val, l1_ratio=l1_ratio)
    clf.fit(X_train, y_train)
    train_errs[i] = np.mean((y_train - clf.predict(X_train)) ** 2)
    test_errs[i] = np.mean((y_test - clf.predict(X_test)) ** 2)
    sparsity[i] = (clf.coef_ != 0.).sum()  # number of non-zero elements in clf.coef_

fig, ax = plt.subplots()
ax.semilogx(l1_ratios, test_errs, label='Test')
ax.semilogx(l1_ratios, train_errs, label='Train')
ax.set_xlabel("l1_ratio")
ax.set_ylabel("Error")
ax.legend()
plt.title('Noisy Data')
plt.show()

fig, ax = plt.subplots()
ax.semilogx(l1_ratios, sparsity)
ax.set_xlabel('l1_ratio')
ax.set_ylabel(r'$||w||_0$')
plt.title('Noisy Data')
plt.show()

# What happens when alpha becomes too big?
# We weight to much the regularization term, meaning that we are searching for a solution that has a very little
# influence from the training data. Therefore the training (and testing) error increases.
train_errs = np.zeros(len(alphas))
test_errs = np.zeros_like(train_errs)
sparsity = np.zeros_like(train_errs)
for i, alpha in enumerate(alphas):
    clf = ElasticNet(alpha=alpha, l1_ratio=l1_val, max_iter=4000)
    clf.fit(X_train, y_train)
    train_errs[i] = np.mean((y_train - clf.predict(X_train)) ** 2)
    test_errs[i] = np.mean((y_test - clf.predict(X_test)) ** 2)
    sparsity[i] = (clf.coef_ != 0.).sum()  # number of non-zero elements in clf.coef_

fig, ax = plt.subplots()
ax.semilogx(alphas, test_errs, label='Test')
ax.semilogx(alphas, train_errs, label='Train')
ax.set_xlabel("alpha")
ax.set_ylabel("Error")
ax.legend()
plt.title('Noisy Data')
plt.show()

fig, ax = plt.subplots()
ax.semilogx(alphas, sparsity)
ax.set_xlabel('alpha')
ax.set_ylabel(r'$||w||_0$')
plt.title('Noisy Data')
plt.show()

# Bonus sub-part:
#
# An alternative formulation/parametrization of the ElasticNet objective is:
# 1/(2*n_samples)*Norm^2{y - X*beta}_2  + alpha*Norm{beta}_1 + beta/2*Norm^2{beta}_2
#
# We can control the L1 and L2 penalty separately. This is equivalent to: a * L1 + b * L2
# Alpha and l1_ratio as functions of a and b, where alpha = a + b and l1_ratio = a / (a + b).
#
# For a fixed value of a, fit the model with increasing values of b. How is the sparsity of the solutions affected
a = 0.005
b_values = np.geomspace(1e-5, 1e2, num=100)
train_errs = np.zeros(len(b_values))
test_errs = np.zeros_like(train_errs)
sparsity = np.zeros_like(train_errs)
for i, b in enumerate(b_values):
    clf = ElasticNet(alpha=a + b, l1_ratio=a / (a + b), max_iter=4000)
    clf.fit(X_train, y_train)
    train_errs[i] = np.mean((y_train - clf.predict(X_train)) ** 2)
    test_errs[i] = np.mean((y_test - clf.predict(X_test)) ** 2)
    sparsity[i] = (clf.coef_ != 0.).sum()

fig, ax = plt.subplots()
ax.semilogx(b_values, test_errs, label='Test')
ax.semilogx(b_values, train_errs, label='Train')
ax.set_xlabel("b value")
ax.set_ylabel("Error")
ax.legend()
plt.show()

fig, ax = plt.subplots()
ax.semilogx(b_values, sparsity)
ax.set_xlabel('a value')
ax.set_ylabel(r'$||w||_0$')
plt.title('Noisy Data')
plt.show()

# L1 ratio
fig, ax = plt.subplots()
ax.semilogx(a / (b_values + a), test_errs, label='Test')
ax.semilogx(a / (b_values + a), train_errs, label='Train')
ax.set_xlabel("l1_ratio")
ax.set_ylabel("Error")
ax.legend()
plt.show()

fig, ax = plt.subplots()
ax.semilogx(a / (b_values + a), sparsity)
ax.set_xlabel('l1_ratio')
ax.set_ylabel(r'$||w||_0$')
plt.show()

# Alpha
fig, ax = plt.subplots()
ax.semilogx(b_values + a, test_errs, label='Test')
ax.semilogx(b_values + a, train_errs, label='Train')
ax.set_xlabel("alpha")
ax.set_ylabel("Error")
ax.legend()
plt.show()

fig, ax = plt.subplots()
ax.semilogx(b_values + a, sparsity)
ax.set_xlabel('alpha')
ax.set_ylabel(r'$||w||_0$')
plt.show()

# Now lets fix b and change a
b = 0.005
a_values = np.geomspace(1e-6, 1e2, num=100)
train_errs = np.zeros(len(a_values))
test_errs = np.zeros_like(train_errs)
sparsity = np.zeros_like(train_errs)
for i, a in enumerate(a_values):
    clf = ElasticNet(alpha=a + b, l1_ratio=a / (a + b), max_iter=4000)
    clf.fit(X_train, y_train)
    train_errs[i] = np.mean((y_train - clf.predict(X_train)) ** 2)
    test_errs[i] = np.mean((y_test - clf.predict(X_test)) ** 2)
    sparsity[i] = (clf.coef_ != 0.).sum()

fig, ax = plt.subplots()
ax.semilogx(a_values, test_errs, label='Test')
ax.semilogx(a_values, train_errs, label='Train')
ax.set_xlabel("a value")
ax.set_ylabel("Error")
ax.legend()
plt.show()

fig, ax = plt.subplots()
ax.semilogx(a_values, sparsity)
ax.set_xlabel('a value')
ax.set_ylabel(r'$||w||_0$')
plt.title('Noisy Data')
plt.show()

# L1 ratio
fig, ax = plt.subplots()
ax.semilogx(a_values / (a_values + b), test_errs, label='Test')
ax.semilogx(a_values / (a_values + b), train_errs, label='Train')
ax.set_xlabel("l1_ratio")
ax.set_ylabel("Error")
ax.legend()
plt.show()

fig, ax = plt.subplots()
ax.semilogx(a_values / (a_values + b), sparsity)
ax.set_xlabel('l1_ratio')
ax.set_ylabel(r'$||w||_0$')
plt.show()

# Alpha
fig, ax = plt.subplots()
ax.semilogx(a_values + b, test_errs, label='Test')
ax.semilogx(a_values + b, train_errs, label='Train')
ax.set_xlabel("alpha")
ax.set_ylabel("Error")
ax.legend()
plt.show()

fig, ax = plt.subplots()
ax.semilogx(a_values + b, sparsity)
ax.set_xlabel('alpha')
ax.set_ylabel(r'$||w||_0$')
plt.show()
# Sparsity exponentially increases with the b value, i.e. with the impact of L2 regularization wrt to L1.

# ## Influence of dataset size
#
# Observe that the datafitting term is normalized by n_samples, hence it should not grow when the dataset becomes taller
# (n_features is fixed, n_samples increases).
#
# Vary one of n_samples, n_features and n_informative_features to observe their influence on the model. What happens
# when n_samples becomes greater that n_informative_features ?
n_samples = 400  # 100 --> I have increased this!
n_features = 200  # 200
n_informative_features = 50  # 50

X_train, X_test, y_train, y_test = train_test_data(n_samples, n_features, n_informative_features,
                                                   noise_level=0.5)

clf = ElasticNet(alpha=0.0, l1_ratio=0.)
clf.fit(X_train, y_train)
print("50 first coefficients of estimated w:\n", clf.coef_[:50])
print("Intercept: %f" % clf.intercept_)
print("Non-zero coefficients: %d out of %d" % ((clf.coef_ != 0.).sum(), clf.coef_.shape[0]))
print("Training error: %.4f percent" % (np.mean((y_train - clf.predict(X_train)) ** 2) * 100))
print("Testing error: %.4f percent" % (np.mean((y_test - clf.predict(X_test)) ** 2) * 100))

# ## Parameter selection with cross validation
#
# In the next section, we use scikit-learn's built in functions to perform cross validated selection of alpha and
# l1_ratio.
X_train, X_test, y_train, y_test = train_test_data(n_samples=100, n_features=300, n_informative_features=20,
                                                   noise_level=0.1)
# using 3-fold cross validation
clf = ElasticNetCV(l1_ratio=[.1, .4, .8, .99, ], cv=3)
clf.fit(X_train, y_train)

# Displaying mean squared errors for various values of l1_ratio (values are over CV folds, thick black line is average
# over folds)
fig, axarr = plt.subplots(2, 2, figsize=(15, 10))
for i, l1_ratio in enumerate(clf.l1_ratio):
    mse = clf.mse_path_[i]
    alphas = clf.alphas_[i]
    axarr.flat[i].semilogx(alphas, mse, ':')
    axarr.flat[i].semilogx(alphas, mse.mean(axis=-1), 'k',
                           label='Average across the folds', linewidth=2)

    axarr.flat[i].set_xlabel(r'$\alpha$')
    axarr.flat[i].set_ylabel('Mean square error')
    axarr.flat[i].set_title('l1_ratio=%s' % l1_ratio)
plt.show()

print("Optimal values for l1_ratio and alpha: %s, %.2e" % (clf.l1_ratio_, clf.alpha_))

n_infos = np.linspace(20, 300, 30)
opt_l1_ratios = []
opt_alphas = []
for n_info in n_infos:
    X_train, X_test, y_train, y_test = train_test_data(n_samples=100, n_features=300,
                                                       n_informative_features=int(n_info), noise_level=0.1)
    # using 3-fold cross validation
    clf = ElasticNetCV(l1_ratio=np.geomspace(1e-4, 1, num=30), cv=3)
    clf.fit(X_train, y_train)
    opt_l1_ratios.append(clf.l1_ratio_)
    opt_alphas.append(clf.alpha_)

plt.figure()
plt.plot(n_infos, opt_l1_ratios)
plt.xlabel('# important features')
plt.ylabel('optimal L1 ratio')
plt.show()

plt.figure()
plt.plot(n_infos, opt_alphas)
plt.xlabel('# important features')
plt.ylabel('optimal alpha')
plt.show()

# How does the best l1_ratio evolve when n_informative_features increases ? Why ?
# If we have more informative features (less sparsity), then the best value for l1_ratio decreases, meaning that we need
# to give a stronger weight to the L2 regularization than to L1.
