#!/usr/bin/env python
# coding: utf-8

import matplotlib.pyplot as plt
import pandas as pd
import pylab as pl
import numpy as np
from sklearn import linear_model
from mpl_toolkits.mplot3d import Axes3D

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
cdf = df[['ENGINESIZE', 'CYLINDERS', 'FUELCONSUMPTION_COMB', 'FUELCONSUMPTION_CITY', 'FUELCONSUMPTION_HWY', 'CO2EMISSIONS']]
print('\nSelected features (10 samples):')
print(cdf.head(10))

# Now, lets plot the Engine Size vs the Emission, to see how linear is their relation:
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
indip_var = ['ENGINESIZE', 'CYLINDERS', 'FUELCONSUMPTION_COMB']
# indip_var = ['ENGINESIZE', 'CYLINDERS', 'FUELCONSUMPTION_CITY', 'FUELCONSUMPTION_HWY']
dip_var = ['CO2EMISSIONS']
train_x = np.asanyarray(train[indip_var])
train_y = np.asanyarray(train[dip_var])

### MULTIPLE LINEAR REGRESSION MODEL
# Modeling data
regr = linear_model.LinearRegression()
regr.fit(train_x, train_y)  # Finds coefficients with Ordinary Least Squares (OLS) method
# The coefficients
print('\nLinear Regression coefficients are:\n')
coeff_df = pd.DataFrame(regr.coef_.T, [indip_var], columns=['Coefficient'])
print(coeff_df)
print('\nIntercept = ', regr.intercept_[0])

# Plot the training data distribution and the linear regression curve
fig = plt.figure()
ax = plt.axes(projection='3d')
# ax.scatter3D(train_x[:, 0], train_x[:, 1], train_y[:, 0], c=train_y[:, 0], cmap='viridis')
ax.scatter3D(train_x[:, 0], train_x[:, 1], train_x[:, 2], c=train_y[:, 0], cmap='viridis')
# curve = regr.coef_[0][:1] * train_x[:, :1] + regr.intercept_[0]
# ax.plot3D(train_x[:, 0], train_x[:, 1], curve[:, 0], '-r')
plt.show()

### EVALUATION OF PREDICTION
y_hat = regr.predict(test[indip_var])
x = np.asanyarray(test[indip_var])
y = np.asanyarray(test[dip_var])
pred_act = pd.DataFrame({'Actual': y[:, 0], 'Predicted': y_hat[:, 0]})
print('\nCheck prediction on first 10 samples:')
print(pred_act.head(10))

# Let's compare different evaluation metrics
print('\nEvaluation metrics compared:')
print("Residual sum of squares (MSE) = %.2f" % np.mean((y_hat - y) ** 2))
print('Variance score = %.2f' % regr.score(x, y))  # Explained variance score: 1 is perfect prediction
