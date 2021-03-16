#!/usr/bin/env python
# coding: utf-8

# Here you will learn a popular machine learning algorithm: Decision Tree. You will use this classification algorithm
# to build a model from historical data of patients, and their response to different medications. Then you use the
# trained decision tree to predict the class of a unknown patient, or to find a proper drug for a new patient.

import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.externals.six import StringIO
import pydotplus
import matplotlib.image as mpimg

### IMPORT AND EXPLORE DATA
# Imagine that you are a medical researcher compiling data for a study. You have collected data about a set of patients,
# all of whom suffered from the same illness. During their course of treatment, each patient responded to one of 5
# medications: Drug A, Drug B, Drug c, Drug x and y. Part of your job is to build a model to find out which drug might
# be appropriate for a future patient with the same illness.
# Import drug200.csv dataset, previously downloaded (at ./Data/) from the IBM Object Storage, particularly from:
# https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/ML0101ENv3/labs/drug200.csv
df = pd.read_csv("./Data/drug200.csv", delimiter=",")
# df = pd.read_csv("/home/simonet/PycharmProjects/PyLearn/7_ScikitLearn-Scipy/IBM_MLcourse/2_Classification/Data/drug200.csv", delimiter=",")

pd.set_option("display.max_columns", None, 'display.width', None)
print('\nData structure:')
print(df.head())  # take a look at the dataset
# The feature set of this dataset is: Age, Sex, Blood Pressure, and Cholesterol.
# The target is the Drug that each patient responded to.
# It is a sample of binary classifier, and you can use the training part of the dataset to build a decision tree,
# and then use it to predict the class of an unknown patient, or to prescribe it to a new patient.

### SELECT AND PREPROCESS DATA
# Create a numpy array with all samples corresponding to the features of interest.
X = df[['Age', 'Sex', 'BP', 'Cholesterol', 'Na_to_K']].values

# And the array of labels.
y = df["Drug"]

# As you may figure out, some features in this dataset are categorical such as Sex or BP. Unfortunately, Sklearn
# Decision Trees do not handle categorical variables. But still we can convert these features to numerical values using
# pandas.get_dummies(). This function converts categorical variables into dummy/indicator variables.
# Sex:
le_sex = preprocessing.LabelEncoder()
le_sex.fit(['F', 'M'])
X[:, 1] = le_sex.transform(X[:, 1])
# Blood Pressure:
le_BP = preprocessing.LabelEncoder()
le_BP.fit(['LOW', 'NORMAL', 'HIGH'])
X[:, 2] = le_BP.transform(X[:, 2])
# Cholesterol:
le_Chol = preprocessing.LabelEncoder()
le_Chol.fit(['NORMAL', 'HIGH'])
X[:, 3] = le_Chol.transform(X[:, 3])

### CREATE TRAIN AND TEST SET
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=3)

### TRAINING THE MODEL AND PREDICTING
# We will first create an instance of the DecisionTreeClassifier called drugTree. Note that inside of the classifier we
# should specify criterion="entropy" in order to see the information gain of each node.
drugTree = tree.DecisionTreeClassifier(criterion="entropy", max_depth=4)

# Train (on training set)
drugTree.fit(X_train, y_train)

# Predict (on testing set)
y_hat = drugTree.predict(X_test)

### ACCURACY EVALUATION
# Visually check prediction on first 10 samples
pred_act = pd.DataFrame({'Actual': y_test, 'Predicted': y_hat})
print('\nCheck prediction on first 10 samples:')
print(pred_act.head(10))

# Compute accuracy on test-set.
test_acc = metrics.accuracy_score(y_test, y_hat)
print("\nDecisionTrees's Accuracy: ", test_acc)

### VISUALIZATION OF DECISION TREE
dot_data = StringIO()
filename = "drugtree.png"
featureNames = df.columns[0:5]
targetNames = df["Drug"].unique().tolist()
out = tree.export_graphviz(drugTree, feature_names=featureNames, out_file=dot_data, class_names=np.unique(y_train),
                           filled=True, special_characters=True, rotate=False)
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
graph.write_png(filename)
img = mpimg.imread(filename)
plt.figure(figsize=(100, 200))
plt.imshow(img, interpolation='nearest')
plt.show()
