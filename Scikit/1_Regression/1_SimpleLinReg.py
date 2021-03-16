#!/usr/bin/env python
# coding: utf-8

import matplotlib.pyplot as plt
import pandas as pd
import pylab as pl
import numpy as np
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
cdf = df[['ENGINESIZE', 'CYLINDERS', 'FUELCONSUMPTION_COMB', 'CO2EMISSIONS']]
print('\nSelected features (10 samples):')
print(cdf.head(10))
# we can plot each of these features:
viz = cdf[['CYLINDERS', 'ENGINESIZE', 'CO2EMISSIONS', 'FUELCONSUMPTION_COMB']]
viz.hist()
plt.show()
# Now, lets plot each of these features vs the Emission, to see how linear is their relation:
plt.figure(figsize=(12, 4))
plt.suptitle('CO2 Emission vs Feature "X"')
plt.subplot(131)
plt.scatter(cdf.FUELCONSUMPTION_COMB, cdf.CO2EMISSIONS, color='blue')
plt.xlabel("Fuel Consumption")
plt.ylabel("Emission")
plt.subplot(132)
plt.scatter(cdf.ENGINESIZE, cdf.CO2EMISSIONS, color='blue')
plt.xlabel("Engine size")
plt.ylabel("Emission")
plt.subplot(133)
plt.scatter(cdf.CYLINDERS, cdf.CO2EMISSIONS, color='blue')
plt.xlabel("Cylinders")
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

### SIMPLE LINEAR REGRESSION MODEL
# Modeling data
regr = linear_model.LinearRegression()
regr.fit(train_x, train_y)
# The coefficients
print('\nLinear Regression coefficients are:')
print('Slope = ', regr.coef_[0][0])
print('Intercept = ', regr.intercept_[0])

# Plot the training data distribution and the linear regression curve
plt.figure()
plt.scatter(train.ENGINESIZE, train.CO2EMISSIONS, color='blue')
plt.plot(train_x, regr.coef_[0][0] * train_x + regr.intercept_[0], '-r')
plt.xlabel("Engine size")
plt.ylabel("Emission")
plt.show()

### EVALUATION OF PREDICTION
test_x = np.asanyarray(test[indip_var])  # input
test_y = np.asanyarray(test[dip_var])  # actual output
test_y_hat = regr.predict(test_x)  # predicted output
pred_act = pd.DataFrame({'Actual': test_y[:, 0], 'Predicted': test_y_hat[:, 0]})
print('\nCheck prediction on first 10 samples:')
print(pred_act.head(10))

# Let's compare different evaluation metrics
print('\nEvaluation metrics compared:')
print("Mean absolute error = %.2f" % np.mean(np.absolute(test_y_hat - test_y)))  # Mean absolute error
print("Residual sum of squares (MSE) = %.2f" % np.mean((test_y_hat - test_y) ** 2))  # Mean Squared Error (MSE)
print("R2-score = %.2f" % r2_score(test_y_hat, test_y))  # R-squared is not an error, but a popular metric for accuracy
