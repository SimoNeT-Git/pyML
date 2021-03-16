#!/usr/bin/env python
# coding: utf-8

import matplotlib.pyplot as plt
import pandas as pd
import pylab as pl
import numpy as np
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score

### IMPORT AND EXPLORE DATA
# Import China GDP dataset (cina_gdp.csv) which contains China's Gross Domestic Product from 1960 to 2014, i.e. the
# market value of all final goods and services produced by China in each year of that period. Such dataset has been
# previously downloaded (at ./Data/) from the IBM Object Storage, particularly from:
# https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/ML0101ENv3/labs/china_gdp.csv
df = pd.read_csv("./Data/china_gdp.csv")
# df = pd.read_csv("/home/simonet/PycharmProjects/PyLearn/7_ScikitLearn-Scipy/IBM_MLcourse/1_Regression/Data/china_gdp.csv")

pd.set_option("display.max_columns", None, 'display.width', None)
print('\nData structure:')
print(df.head())  # take a look at the dataset
# Lets have a descriptive exploration on our data.
print('\nData description:')
print(df.describe())  # summarize the data

# Lets select interesting features to explore more.
cdf = df[['Year', 'Value']]
# Lets plot China GDP value for each Year:
plt.figure(figsize=(8, 5))
x_data, y_data = (cdf.Year, cdf.Value)  # or, equivalently: (df["Year"].values, df["Value"].values)
plt.plot(x_data, y_data, 'ro')
plt.ylabel('China GDP')
plt.xlabel('Year')
plt.show()

### CREATE TRAIN AND TEST SET
# Lets first of all normalize our data.
xdata = x_data / max(x_data)
ydata = y_data / max(y_data)
# Lets split our dataset into train and test sets, 80% of the entire data for training, and the 20% for testing.
# We create a mask to select random rows using np.random.rand() function
msk = np.random.rand(len(df)) < 0.8
train_x = xdata[msk]
train_y = ydata[msk]
test_x = xdata[~msk]
test_y = ydata[~msk]

### NON-LINEAR REGRESSION MODEL
# From an initial look at the plot, we determine that the logistic function could be a good approximation, since it has
# the property of starting with a slow growth, increasing growth in the middle, and then decreasing again at the end.
# Now, let's build our regression model and initialize its parameters.
def sigmoid(x, Beta_1, Beta_2):
    y = 1 / (1 + np.exp(-Beta_1 * (x - Beta_2)))
    return y


# Our task here is to find the best parameters for such model. To this purpose we use the curve_fit function specifying
# here the type of function to use for approximating the data.
popt, pcov = curve_fit(sigmoid, train_x, train_y)  # build the model using train set
test_y_hat = sigmoid(test_x, *popt)  # predict using test set
# The coefficients
print('\nNon-linear Regression coefficients are:')
print("beta_1 = %f, beta_2 = %f" % (*popt,))  # *popt is equal to popt[0] and popt[1]

# Plot the training data distribution and the non-linear regression curve.
x = np.linspace(1960, 2015, 55)
x = x / max(x)
y = sigmoid(x, *popt)
plt.figure(figsize=(8, 5))
plt.plot(train_x, train_y, 'ro', label='data')
plt.plot(x, y, linewidth=3.0, label='fit')
plt.legend(loc='best')
plt.ylabel('GDP')
plt.xlabel('Year')
plt.show()

### EVALUATION OF PREDICTION
pred_act = pd.DataFrame({'Actual': test_y, 'Predicted': np.round(test_y_hat, 6)})
print('\nCheck prediction on first 10 samples:')
print(pred_act.head(10))

# Let's compare different evaluation metrics
print('\nEvaluation metrics compared:')
print("Mean absolute error: %.2f" % np.mean(np.absolute(test_y_hat - test_y)))
print("Residual sum of squares (MSE): %.2f" % np.mean((test_y_hat - test_y) ** 2))
print("R2-score: %.2f" % r2_score(test_y_hat, test_y))
