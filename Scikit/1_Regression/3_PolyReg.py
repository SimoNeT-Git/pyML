#!/usr/bin/env python
# coding: utf-8

import matplotlib.pyplot as plt
import pandas as pd
import pylab as pl
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn import linear_model
from sklearn.metrics import r2_score

### IMPORT AND EXPLORE DATA
# Import fuel consumption dataset (FuelConsumptionCo2.csv) which contains model-specific fuel consumption ratings and
# estimated carbon dioxide emissions for new light-duty vehicles for retail sale in Canada. Such dataset has been
# previously downloaded (at ./Data/) from the IBM Object Storage, particularly from:
# https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/ML0101ENv3/labs/FuelConsumptionCo2.csv
df = pd.read_csv("./Data/FuelConsumptionCo2.csv")
# df = pd.read_csv("/home/simonet/PycharmProjects/PyLearn/7_ScikitLearn-Scipy/IBM_MLcourse/1_Regression/Data/FuelConsumptionCo2.csv")

pd.set_option("display.max_columns", None, 'display.width', None)
print('\nData structure:')
print(df.head())  # take a look at the dataset
# Lets have a descriptive exploration on our data.
print('\nData description:')
print(df.describe())  # summarize the data

# Lets select some features to explore more.
cdf = df[
    ['ENGINESIZE', 'CYLINDERS', 'FUELCONSUMPTION_COMB', 'FUELCONSUMPTION_CITY', 'FUELCONSUMPTION_HWY', 'CO2EMISSIONS']]
print('\nSelected features (10 samples):')
print(cdf.head(10))
# Lets plot Emission values with respect to Engine size:
plt.figure()
plt.scatter(cdf.ENGINESIZE, cdf.CO2EMISSIONS, color='blue')
plt.xlabel("Engine size")
plt.ylabel("Emission")
plt.show()

### CREATE TRAIN AND TEST SET
# Lets split our dataset into train and test sets, 80% of the entire data for training, and the 20% for testing.
# We create a mask to select random rows using np.random.rand() function
msk = np.random.rand(len(df)) < 0.8
train = cdf[msk]
test = cdf[~msk]
indip_var = ['ENGINESIZE']
dip_var = ['CO2EMISSIONS']
train_x = np.asanyarray(train[indip_var])
train_y = np.asanyarray(train[dip_var])
test_x = np.asanyarray(test[indip_var])
test_y = np.asanyarray(test[dip_var])

### POLYNOMIAL REGRESSION MODEL
# Lets first generate an array of all polynomial combinations (with degree less than or equal to the specified degree)
# of the features in the training set. In this case train_x has 1 feature, thus train_x_poly will have 3 features, i.e
# for degree=0, degree=1 and degree=2 (since we set the degree of our polynomial to 2).
# In other words, fit_transform takes our x values (samples of our feature), and outputs a list of our data raised from
# power of 0 to power of 2.
degree = 2
poly = PolynomialFeatures(degree=degree)
train_x_poly = poly.fit_transform(train_x)


def polynom(x, deg, theta, c):
    y = c
    for k in range(deg):
        y += theta[k + 1] * np.power(x, k + 1)
    return y


# Now, we can deal with it as 'linear regression' problem: polynomial regression is considered to be a special case of
# traditional multiple linear regression.
regr = linear_model.LinearRegression()
regr.fit(train_x_poly, train_y)
# The coefficients
print('\nPolynomial Regression coefficients are:')
indip_var_poly = ['1', 'x'] + ['x^'+str(exp+2) for exp in range(degree-1)]
coeff_df = pd.DataFrame(regr.coef_.T, [indip_var_poly], columns=['Coefficient'])
print(coeff_df)
print('\nIntercept = ', regr.intercept_[0])

# Plot the training data distribution and the polynomial regression curve.
plt.figure()
plt.scatter(train.ENGINESIZE, train.CO2EMISSIONS, color='blue')
XX = np.arange(0.0, 10.0, 0.1)
yy = polynom(XX, degree, regr.coef_[0], regr.intercept_[0])
plt.plot(XX, yy, '-r')
plt.xlabel("Engine size")
plt.ylabel("Emission")
plt.show()

### EVALUATION OF PREDICTION
test_x_poly = poly.fit_transform(test_x)  # input
test_y_hat = regr.predict(test_x_poly)  # predicted output (actual output is test_y)
pred_act = pd.DataFrame({'Actual': test_y[:, 0], 'Predicted': test_y_hat[:, 0]})
print('\nCheck prediction on first 10 samples:')
print(pred_act.head(10))

# Let's compare different evaluation metrics
print('\nEvaluation metrics compared:')
print("Mean absolute error = %.2f" % np.mean(np.absolute(test_y_hat - test_y)))
print("Residual sum of squares (MSE) = %.2f" % np.mean((test_y_hat - test_y) ** 2))
print("R2-score = %.2f" % r2_score(test_y_hat, test_y))
