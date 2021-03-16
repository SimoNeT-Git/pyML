# In this notebook we try to practice all the CLASSIFICATION algorithms that we learned in this course.
# We load a dataset using Pandas library, and apply the following algorithms, and find the best one for this specific
# dataset by accuracy evaluation methods.

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import preprocessing
import seaborn as sns

figshow = False

### IMPORT DATASET (Note: it is the Train set only!)
# This dataset is about past loans. The Loan_train.csv data set includes details of 346 customers whose loan are already
# paid off or defaulted. It includes following fields:
# | Field          | Description                                                                           |
# |----------------|---------------------------------------------------------------------------------------|
# | Unnamed: 0     | Integer values (don't know the meaning)                                               |
# | Unnamed: 1     | Integer values (don't know the meaning)                                               |
# | Loan_status    | Whether a loan is paid off on in collection                                           |
# | Principal      | Basic principal loan amount                                                           |
# | Terms          | Origination terms which can be weekly (7 days), biweekly, and monthly payoff schedule |
# | Effective_date | When the loan got originated and took effects                                         |
# | Due_date       | Since it’s one-time payoff schedule, each loan has one single due date                |
# | Age            | Age of applicant                                                                      |
# | Education      | Education of applicant                                                                |
# | Gender         | The gender of applicant                                                               |
# Import loan_train.csv dataset, previously downloaded (at ./Data/) from the IBM Object Storage, from:
# https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/ML0101ENv3/labs/loan_train.csv
df = pd.read_csv("./Data/loan_train.csv")
# df = pd.read_csv("/home/simonet/PycharmProjects/PyLearn/7_ScikitLearn-Scipy/IBM_MLcourse/Project/Data/loan_train.csv")

### VISUALIZE AND EXPLORE DATASET
df = df[list(df.columns[2:])]  # remove unnamed columns, i.e leave ['loan_status', 'Principal', 'terms',
# 'effective_date', 'due_date', 'age', 'education', 'Gender']

pd.set_option("display.max_columns", None, 'display.width', None)
print('\nData structure:')
print(df.head())
print('\nData description:')
print(df.describe())

# Plot some histograms from the data
df.hist()
if figshow:
    plt.show()

### PRE-PROCESS DATA AND Feature/Label SETS CREATION
## Pre-processing
# Change dates format
df.loc[:, 'due_date'] = pd.to_datetime(df.loc[:, 'due_date'])
df.loc[:, 'effective_date'] = pd.to_datetime(df.loc[:, 'effective_date'])

# Add 'dayofweek' column representing the time of loan arrival
df['dayofweek'] = df['effective_date'].dt.dayofweek
# To better represent the time of loan arrival lets use Feature binarization to set a threshold value, e.g. day 3, so
# that in the new column "weekend" we will have 1 if the applicant received the loan after day 3 and 0 otherwise
th = 4
df['weekend'] = df['dayofweek'].apply(lambda x: 1 if (x > th) else 0)

# Add 'noticeweek' column representing the number of weeks of notice anticipation
df['noticeweek'] = df['due_date'].dt.weekofyear - df['effective_date'].dt.weekofyear
# As before, lets now set a threshold for the number of weeks the notice arrived before its payment
th_notice = 4
df['notice'] = df['noticeweek'].apply(lambda x: 1 if (x > th_notice) else 0)

##########################################################################
# Let’s see how many applicants payed the loan and how many didn't
perc = df['loan_status'].value_counts()
print("\nNumber of applicants who paid off the loan in time vs those who didn't (and have gone into collection):")
print(perc)
# Note: 260 people have paid off the loan on time while 86 have gone into collection

# Lets look at the influence of applicants' gender on loan payment
print("\nDistinguishing between applicant's gender, we display the percentage of them who did and didn't "
      "pay off the loan:")
print(df.groupby(['Gender'])['loan_status'].value_counts(normalize=True) * 100)
# Note: 86 % of female pay there loans while only 73 % of males pay there loan

# Lets look at the influence of applicants' education on loan payment
print("\nDistinguishing between applicant's education, we display the percentage of them who did and didn't "
      "pay off the loan:")
print(df.groupby(['education'])['loan_status'].value_counts(normalize=True) * 100)
# Note: education level seems to have no influence...

# Lets look at the time when loan arrived (since, from the previous plots, we saw that people who get the loan at the
# end of the week don't pay it off):
print("\nDistinguishing between applicants who received the loan during the first " + str(th) + " days of the week (0) "
      "and those who received it later (1),\nwe display the percentage of them who did (PAIDOFF) and didn't "
      "(COLLECTION) pay off the loan:")
print(df.groupby(['weekend'])['loan_status'].value_counts(normalize=True) * 100)
# Note: 95 % of people who received the loan during the first 4 days of the week payed the loan, while only 59 % of
# those who received it later payed the loan

# Lets look at the number of weeks of loan notification:
print("\nDistinguishing between applicants who received the loan with less than " + str(th_notice) + " weeks in advance "
      "(0) and those who received it with more advance (1),\nwe display the percentage of them who did (PAIDOFF) and"
      "didn't (COLLECTION) pay off the loan:")
print(df.groupby(['notice'])['loan_status'].value_counts(normalize=True) * 100)
# Note: 82 % of people who received the loan with less than 4 weeks of advance payed their loan, while only 59 % of
# those who received it with more advance payed it

# Lets plot some histograms to understand data better:
# First, we plot n° of applicants (distinguishing between gender and loan_status) for each principal (i.e. loan value)
bins = np.linspace(df.Principal.min(), df.Principal.max(), 10)
g = sns.FacetGrid(df, col="Gender", hue="loan_status", palette="Set1", col_wrap=2)
g.map(plt.hist, 'Principal', bins=bins, ec="k")
g.axes[-1].legend()
if figshow:
    plt.show()
# Then, we plot n° of applicants (distinguishing between gender and loan_status) with respect to their age
bins = np.linspace(df.age.min(), df.age.max(), 10)
g = sns.FacetGrid(df, col="Gender", hue="loan_status", palette="Set1", col_wrap=2)
g.map(plt.hist, 'age', bins=bins, ec="k")
g.axes[-1].legend()
if figshow:
    plt.show()

# Now, lets look at the n° of applicants (distinguishing between gender and loan_status) with respect to the day of the
# week they get the loan
bins = np.linspace(df.dayofweek.min(), df.dayofweek.max(), 10)
g = sns.FacetGrid(df, col="Gender", hue="loan_status", palette="Set1", col_wrap=2)
g.map(plt.hist, 'dayofweek', bins=bins, ec="k")
g.axes[-1].legend()
if figshow:
    plt.show()

# Finally, lets look at the n° of applicants (distinguishing between gender and loan_status) with respect to the number
# of weeks of notice anticipation
bins = np.linspace(df.dayofweek.min(), df.dayofweek.max(), 10)
g = sns.FacetGrid(df, col="Gender", hue="loan_status", palette="Set1", col_wrap=2)
g.map(plt.hist, 'noticeweek', bins=bins, ec="k")
g.axes[-1].legend()
if figshow:
    plt.show()
##########################################################################

## Convert Categorical features to numerical values:
# First, lets convert GENDER colmun
df['Gender'].replace(to_replace=['male', 'female'], value=[0, 1], inplace=True)  # 0 for male, 1 for female

# Create Feature DataFrame
Feature = df[['Principal', 'terms', 'age', 'Gender', 'weekend', 'notice']]

# Convert EDUCATION column: here lets use One Hot encoding technique to convert categorical variables to binary and
# append them to the Feature DataFrame, i.e. each category (e.g. High School or Below, Bachelor, ...) in the Education
# column of the dataset becomes a new column in the Feature set where values are 1 (if it is the education level of
# that person) or 0 (if not)
#Feature = pd.concat([Feature, pd.get_dummies(df['education'])], axis=1)
#Feature.drop(['Master or Above'], axis=1, inplace=True)
print("\n\nNote: I CHOSE NOT TO CONSIDER EDUCATION BECAUSE FROM THE PERCENTAGE OF PEOPLE WHO DO ("+
      str(round(perc[0]*100, 0))+"%) AND DON'T ("+str(round(perc[1]*100, 0))+"%)\n"
      "PAY THEIR LONE, IT SEEMS THAT EDUCATION DOES NOT INFLUENCE. In fact we see the same percentages for all levels\n"
      "of education (except for 'Master or Above' where we see a 50%/50% ratio, but in this group there is a\n"
      "significantly lower number of people).\n\n")

# Feature and Label arrays creation
X = Feature
y = df['loan_status']
y.replace(to_replace=['COLLECTION', 'PAIDOFF'], value=[0, 1], inplace=True)

# Display features and labels
print('\nFeature set:')
print(X.head())
print('\nLabels:')
print(y.head())

## Normalize Data:
# Data Standardization give data zero mean and unit variance (technically should be done after train-test split)
X = preprocessing.StandardScaler().fit(X).transform(X)
y = y.values  # make y a numpy array

# Now, it is your turn, use the training set to build an accurate model. Then use the test set to report the accuracy of
# the model. You should use the following algorithm:
# - K Nearest Neighbor(KNN)
# - Decision Tree
# - Support Vector Machine
# - Logistic Regression
#
#
# Notice:
# - To make a better model, you can go above and change the pre-processing, feature selection, feature-extraction, ...
# - You should use either scikit-learn, Scipy or Numpy libraries for developing the classification algorithms.
# - You should include the code of the algorithm in the following cells.


### IMPORT TEST SET
# Lets import the test set for future evaluation of classification models applied to the training set.
# Import loan_test.csv, previously downloaded (at ./Data/) from the IBM Object Storage, i.e. from:
# https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/ML0101ENv3/labs/loan_test.csv
test_df = pd.read_csv('./Data/loan_test.csv')
# test_df = pd.read_csv("/home/simonet/PycharmProjects/PyLearn/7_ScikitLearn-Scipy/IBM_MLcourse/Project/Data/loan_test.csv")

test_df = test_df[list(test_df.columns[2:])]
test_df.loc[:, 'due_date'] = pd.to_datetime(test_df.loc[:, 'due_date'])  # change date format
test_df.loc[:, 'effective_date'] = pd.to_datetime(test_df.loc[:, 'effective_date'])  # change date format
test_df['dayofweek'] = test_df['effective_date'].dt.dayofweek
test_df['weekend'] = test_df['dayofweek'].apply(lambda x: 1 if (x > th) else 0)
test_df['noticeweek'] = test_df['due_date'].dt.weekofyear - test_df['effective_date'].dt.weekofyear
test_df['notice'] = test_df['noticeweek'].apply(lambda x: 1 if (x > th_notice) else 0)
test_df['Gender'].replace(to_replace=['male', 'female'], value=[0, 1], inplace=True)  # 0 for male, 1 for female
Feature_test = test_df[['Principal', 'terms', 'age', 'Gender', 'weekend', 'notice']]
# Feature_test = pd.concat([Feature_test, pd.get_dummies(test_df['education'])], axis=1)
# Feature_test.drop(['Master or Above'], axis=1, inplace=True)

X_test = Feature_test
X_test = preprocessing.StandardScaler().fit(X_test).transform(X_test)
y_test = test_df['loan_status']
y_test.replace(to_replace=['COLLECTION', 'PAIDOFF'], value=[0, 1], inplace=True)
y_test = y_test.values

### K Nearest Neighbor(KNN)
# Notice: You should find the best k to build the model with the best accuracy.  
# **warning:** You should not use the loan_test.csv for finding the best k, however, you can split your train_loan.csv
# into train and test to find the best k.
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics

# Lets find the best k value
X_trKNN, X_teKNN, y_trKNN, y_teKNN = train_test_split(X, y, test_size=0.2, random_state=1)
Ks = 21
mean_jacc = np.zeros(Ks - 1)
mean_f1 = np.zeros(Ks - 1)
std_acc = np.zeros(Ks - 1)
for k in range(1, Ks):
    # Train Model and Predict
    neigh = KNeighborsClassifier(n_neighbors=k).fit(X_trKNN, y_trKNN)
    yhat = neigh.predict(X_teKNN)
    mean_jacc[k - 1] = metrics.accuracy_score(y_teKNN, yhat)
    mean_f1[k - 1] = metrics.f1_score(y_teKNN, yhat)
    std_acc[k - 1] = np.std(yhat == y_teKNN) / np.sqrt(yhat.shape[0])
# Plot model accuracy for different number of neighbors
plt.figure()
plt.plot(range(1, Ks), mean_jacc, 'g')
plt.fill_between(range(1, Ks), mean_jacc - std_acc, mean_jacc + std_acc, alpha=0.10)
plt.plot(range(1, Ks), mean_f1, 'r')
plt.fill_between(range(1, Ks), mean_f1 - std_acc, mean_f1 + std_acc, alpha=0.10)
plt.legend(('Jaccard', 'f1-score', '+/-3std', '+/-3std'))
plt.ylabel('Accuracy')
plt.xlabel('Number of Neighbors K')
plt.tight_layout()
if figshow:
    plt.show()
# Save best K value and print info
mean_acc = (mean_jacc + mean_f1) / 2
bestK = mean_acc.argmax() + 1
print("\nThe best k value for KNN model is", bestK)

# Lets train the model with the best k value
KNN = KNeighborsClassifier(n_neighbors=bestK, algorithm='auto')  # , algorithm='auto'
KNN.fit(X, y)

# Lets test it making predictions
yhat_KNN = KNN.predict(X_test)

### Decision Tree
from sklearn import tree

# Lets train the model
DecTree = tree.DecisionTreeClassifier(criterion="entropy", max_depth=4)
DecTree.fit(X, y)

# Lets test it making predictions
yhat_DT = DecTree.predict(X_test)

### Logistic Regression
from sklearn.linear_model import LogisticRegression

# Lets train the model
LR = LogisticRegression(C=1, solver='liblinear', penalty='l2')
LR.fit(X, y)

# Lets test it making predictions
yhat_LR = LR.predict(X_test)
yhat_prob_LR = LR.predict_proba(X_test)

### Support Vector Machine
from sklearn import svm

# Lets train the model
SVM = svm.SVC(C=1, kernel='rbf', probability=True)  # --> we can define logloss by computing SVM.predict_proba(X_test)
SVM.fit(X, y)

# Lets test it making predictions
yhat_SVM = SVM.predict(X_test)
yhat_proba_SVM = SVM.predict_proba(X_test)

### Model Evaluation using Test set
from sklearn.metrics import jaccard_similarity_score, accuracy_score, f1_score, log_loss

# K Nearest Neighbor
jacc_KNN = round(accuracy_score(y_test, yhat_KNN), 2)
f1_KNN = round(f1_score(y_test, yhat_KNN), 2)
ll_KNN = 'NA'

# Decision Tree
jacc_DT = round(accuracy_score(y_test, yhat_DT), 2)
f1_DT = round(f1_score(y_test, yhat_DT), 2)
ll_DT = 'NA'

# Logistic Regression
jacc_LR = round(accuracy_score(y_test, yhat_LR), 2)
f1_LR = round(f1_score(y_test, yhat_LR), 2)
ll_LR = round(log_loss(y_test, yhat_prob_LR), 2)

# Support Vector Machine
jacc_SVM = round(accuracy_score(y_test, yhat_SVM), 2)
f1_SVM = round(f1_score(y_test, yhat_SVM), 2)
ll_SVM = round(log_loss(y_test, yhat_proba_SVM), 2)

# Print table for comparison
from tabulate import tabulate

table = [["Algorithm", "Jaccard", "F1-score", "LogLoss"],
         ["KNN", jacc_KNN, f1_KNN, ll_KNN],
         ["Decision Tree", jacc_DT, f1_DT, ll_DT],
         ["Logistic Regression", jacc_LR, f1_LR, ll_LR],
         ["SVM", jacc_SVM, f1_SVM, ll_SVM]]
print('\n\n')
print(tabulate(table, headers="firstrow", numalign='center', stralign='center', tablefmt="fancy_grid"))

# You should be able to report the accuracy of the built model using different evaluation metrics:
# | Algorithm          | Jaccard | F1-score | LogLoss |
# |--------------------|---------|----------|---------|
# | KNN                | ?       | ?        | NA      |
# | Decision Tree      | ?       | ?        | NA      |
# | SVM                | ?       | ?        | NA      |
# | LogisticRegression | ?       | ?        | ?       |
