#!/usr/bin/env python
# coding: utf-8

# **K-Nearest Neighbors** is an algorithm for supervised learning. Where the data is 'trained' with data points
# corresponding to their classification. Once a point is to be predicted, it takes into account the 'K' nearest points
# to it to determine it's classification. In this sense, it is important to consider the value of K.

import itertools
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import matplotlib.ticker as ticker
from matplotlib.ticker import NullFormatter
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics

### IMPORT AND EXPLORE DATA
# Imagine a telecommunications provider has segmented its customer base by service usage patterns, categorizing the
# customers into four groups. If demographic data can be used to predict group membership, the company can customize
# offers for individual prospective customers. It is a classification problem. The example focuses on using demographic
# data, such as region, age, and marital, to predict usage patterns.
# Import teleCust1000t.csv dataset, previously downloaded (at ./Data/) from the IBM Object Storage, particularly from:
# https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/ML0101ENv3/labs/teleCust1000t.csv
df = pd.read_csv("./Data/teleCust1000t.csv")
# df = pd.read_csv("/home/simonet/PycharmProjects/PyLearn/7_ScikitLearn-Scipy/IBM_MLcourse/2_Classification/Data/teleCust1000t.csv")

pd.set_option("display.max_columns", None, 'display.width', None)
print('\nData structure:')
print(df.head())  # take a look at the dataset
print('\nThe whole feature set is:\n', list(df.columns))

# The target field, called custcat, has four possible values that correspond to the four customer groups:
# 1- Basic Service; 2- E-Service; 3- Plus Service; 4- Total Service
# Our objective is to build a classifier, to predict the class of unknown cases with K nearest neighbour.
# Let’s see how many of each class is in our data set:
df['custcat'].value_counts()  # 1- 266; 2- 217; 3- 281; 4- 236
# Just as an example of data visualization, let's plot the histogram of income feature.
df.hist(column='income', bins=50)

### SELECT AND NORMALIZE DATA
# First, to use scikit-learn library, we have to convert the Pandas data frame to a Numpy array.
X = df[['region', 'tenure', 'age', 'marital', 'address', 'income', 'ed', 'employ', 'retire', 'gender',
        'reside']].values  # .astype(float)
# Data Standardization give data zero mean and unit variance, it is good practice, especially for algorithms such as
# KNN which is based on distance of cases. So, let's normalize our data:
X = preprocessing.StandardScaler().fit(X).transform(X.astype(float))
# Let's create the array of labels.
y = df['custcat'].values

### CREATE TRAIN AND TEST SET
# Out of Sample Accuracy is the percentage of correct predictions that the model makes on data that the model has
# NOT been trained on. Doing a train and test on the same dataset will most likely have low out-of-sample accuracy,
# due to the likelihood of being over-fit. It is important that our models have a high, out-of-sample accuracy, becaus
# the purpose of any model, of course, is to make correct predictions on unknown data. So how can we improve
# out-of-sample accuracy? One way is to use the Train/Test Split approach. This involves splitting the dataset into
# training and testing sets respectively, which are mutually exclusive. After which, you train with the training set and
# test with the testing set. This will provide a more accurate evaluation on out-of-sample accuracy because the testing
# set is not part of the dataset that has been used for training.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=4)

### TRAINING THE MODEL AND PREDICTING
# Define K
k1 = 4  # Lets start the algorithm with k=4 for now
k2 = 6  # And lets also try with k=6
# Train (on training set)
neigh1 = KNeighborsClassifier(n_neighbors=k1).fit(X_train, y_train)
neigh2 = KNeighborsClassifier(n_neighbors=k2).fit(X_train, y_train)
# Predict (on testing set)
y_hat1 = neigh1.predict(X_test)
y_hat2 = neigh2.predict(X_test)

### ACCURACY EVALUATION
# In multilabel classification, accuracy classification score is a function that computes subset accuracy. This
# function is equal to the jaccard_similarity_score function. Essentially, it calculates how closely the actual labels
# and predicted labels are matched in the test set.
train_acc1 = metrics.accuracy_score(y_train, neigh1.predict(X_train))
test_acc1 = metrics.accuracy_score(y_test, y_hat1)
train_acc2 = metrics.accuracy_score(y_train, neigh2.predict(X_train))
test_acc2 = metrics.accuracy_score(y_test, y_hat2)
print('\nTwo examples of K choices:')
print("- k=" + str(k1) + " yields to a train-set accuracy of", train_acc1,
      "and a test-set accuracy of", test_acc1)
print("- k=" + str(k2) + " yields to a train-set accuracy of", train_acc2,
      "and a test-set accuracy of", test_acc2)

#### CHOOSE THE BEST K VALUE
# K in KNN, is the number of nearest neighbors to examine. It is supposed to be specified by the User. So, how can we
# choose right value for K? The general solution is to train and test (find accuracy) of the model iterating over
# different K values: starting with k=1 and increasing it at every iteration. Then choose K that gives best accuracy.
# Let's try 15 different K values.
Ks = 10
mean_acc = np.zeros((Ks - 1))
std_acc = np.zeros((Ks - 1))
for k in range(1, Ks):
    # Train Model and Predict
    neigh = KNeighborsClassifier(n_neighbors=k).fit(X_train, y_train)
    y_hat = neigh.predict(X_test)
    mean_acc[k - 1] = metrics.accuracy_score(y_test, y_hat)
    std_acc[k - 1] = np.std(y_hat == y_test) / np.sqrt(y_hat.shape[0])

# Plot model accuracy for different number of neighbors
plt.figure()
plt.plot(range(1, Ks), mean_acc, 'g')
plt.fill_between(range(1, Ks), mean_acc - std_acc, mean_acc + std_acc, alpha=0.10)
plt.legend(('Mean', '+/- 3std'))
plt.ylabel('Accuracy')
plt.xlabel('Number of Neighbors K')
plt.tight_layout()
plt.show()
print("\nThe best accuracy equals", mean_acc.max(), "and it is reached with k=" + str(mean_acc.argmax() + 1))
