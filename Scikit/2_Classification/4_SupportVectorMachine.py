#!/usr/bin/env python
# coding: utf-8

# In this notebook, you will use SVM (Support Vector Machines) to build and train a model using human cell records, and
# classify cells to whether the samples are benign or malignant.
# SVM works by mapping data to a high-dimensional feature space so that data points can be categorized, even when the
# data are not otherwise linearly separable. A separator between the categories is found, then the data is transformed
# in such a way that the separator could be drawn as a hyperplane. Following this, characteristics of new data can be
# used to predict the group to which a new record should belong.

import pandas as pd
import pylab as pl
import numpy as np
import scipy.optimize as opt
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import jaccard_score
import itertools
import matplotlib.pyplot as plt

### IMPORT AND EXPLORE DATA
# The example is based on a dataset that is publicly available from the UCI Machine Learning Repository (Asuncion and
# Newman, 2007)[http://mlearn.ics.uci.edu/MLRepository.html]. The dataset consists of several hundred human cell sample
# records, each of which contains the values of a set of cell characteristics. For the purposes of this example, we're
# using a dataset that has a relatively small number of predictors in each record.
# Import cell_samples.csv dataset, previously downloaded (at ./Data/) from the IBM Object Storage, particularly from:
# https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/ML0101ENv3/labs/cell_samples.csv
df = pd.read_csv("./Data/cell_samples.csv")
# df = pd.read_csv("/home/simonet/PycharmProjects/PyLearn/7_ScikitLearn-Scipy/IBM_MLcourse/2_Classification/Data/cell_samples.csv")

pd.set_option("display.max_columns", None, 'display.width', None)
print('\nData structure:')
print(df.head())  # take a look at the dataset

# The ID field contains the patient identifiers. The characteristics of the cell samples from each patient are contained
# in fields Clump to Mit. The values are graded from 1 to 10, with 1 being the closest to benign. The Class field
# contains the diagnosis, as confirmed by separate medical procedures, as to whether the samples are benign (value = 2)
# or malignant (value = 4).
# Lets look at the distribution of the classes based on Clump thickness and Uniformity of cell size:
ax = df[df['Class'] == 4][0:50].plot(kind='scatter', x='Clump', y='UnifSize', color='DarkBlue', label='malignant')
df[df['Class'] == 2][0:50].plot(kind='scatter', x='Clump', y='UnifSize', color='Yellow', label='benign', ax=ax)
plt.show()

# Before pre-processing and selecting data, lets first look at columns data types:
print('\nColumns data types:\n' + str(df.dtypes))

# It looks like the BareNuc column includes some values that are not numerical. We can drop those rows:
cell_df = df[pd.to_numeric(df.loc[:, 'BareNuc'], errors='coerce').notnull()]
cell_df.loc[:, 'BareNuc'] = cell_df.loc[:, 'BareNuc'].astype('int')
print('\nColumns data types:\n' + str(cell_df.dtypes))

### SELECT AND NORMALIZE DATA
# Create a numpy array with all samples corresponding to the features of interest.
# Lets define X, and y for our dataset:
feature_df = cell_df.loc[:,
             ['Clump', 'UnifSize', 'UnifShape', 'MargAdh', 'SingEpiSize', 'BareNuc', 'BlandChrom', 'NormNucl', 'Mit']]
X = np.asarray(feature_df)

# We want the model to predict the value of Class (that is, benign (=2) or malignant (=4)). As this field can have one
# of only two possible values, we need to change its measurement level to reflect this.
cell_df.loc[:, 'Class'] = cell_df.loc[:, 'Class'].astype('int')
y = np.asarray(cell_df['Class'])

### CREATE TRAIN AND TEST SET
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=4)

### TRAINING THE MODEL AND PREDICTING
# The SVM algorithm offers a choice of kernel functions for performing its processing. Basically, mapping data into a
# higher dimensional space is called kernelling. The mathematical function used for the transformation is known as the
# kernel function, and can be of different types, such as:
#     1.Linear
#     2.Polynomial
#     3.Radial basis function (RBF)
#     4.Sigmoid
# Each of these functions has its characteristics, its pros and cons, and its equation, but as there's no easy way of
# knowing which function performs best with any given dataset, we usually choose different functions in turn and compare
# the results. Let's just use the default, RBF (Radial Basis Function) for this lab.
clf = svm.SVC(kernel='rbf')
clf.fit(X_train, y_train)

# After being fitted, the model can then be used to predict new values:
y_hat = clf.predict(X_test)

### ACCURACY EVALUATION
## --> Confusion Matrix and F1 score
# One way of looking at accuracy of classifier is to look at the confusion matrix.
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = np.round(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100, 0)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(int(cm[i, j])) + (' %' if normalize else ''),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


# Compute confusion matrix
cnf_matrix = confusion_matrix(y_test, y_hat, labels=[2, 4])
np.set_printoptions(precision=2)

# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=['Benign(2)', 'Malignant(4)'], normalize=True, title='Confusion matrix')
plt.show()
print('\nClassification Report:\n\n' + classification_report(y_test, y_hat))

## --> Jaccard Index
# Lets try jaccard index for accuracy:
acc_jacc = round(accuracy_score(y_test, y_hat), 2)
jacc_macro = round(jaccard_score(y_test, y_hat, average='macro'), 2)
jacc_weight = round(jaccard_score(y_test, y_hat, average='weighted'), 2)
print('\n                 Jaccard\n                   score\n\n' + '    accuracy        ' +
      str(acc_jacc) + '\n   macro avg        ' + str(jacc_macro) + '\nweighted avg        ' + str(jacc_weight))


### TRY DIFFERENT KERNEL FOR THE SVM MODEL
# Can you rebuild the model, but this time with a linear kernel? You can use kernel='linear' option, when you define the
# svm. How the accuracy changes with the new kernel function?
kernel = 'linear'
clf2 = svm.SVC(kernel=kernel)
clf2.fit(X_train, y_train)
y_hat2 = clf2.predict(X_test)
print("\n\nAn SVM with kernel='"+kernel+"' gives:")
print("Avg F1-score = %.2f" % f1_score(y_test, y_hat2, average='weighted'))
print("Jaccard score = %.2f" % accuracy_score(y_test, y_hat2))
