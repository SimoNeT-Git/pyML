#!/usr/bin/env python
# coding: utf-8

# There are many models for clustering. In this notebook, we will be presenting the model that is considered one of the
# simplest models amongst them. Despite its simplicity, the K-means is vastly used for clustering in many data science
# applications, especially useful if you need to quickly discover insights from unlabeled data. In this notebook, you
# will learn how to use k-Means for customer segmentation.
# Some real-world applications of k-means are: Customer segmentation; Understand what the visitors of a website are
# trying to accomplish; Pattern recognition; Machine learning; Data compression
# In this notebook we practice k-means clustering with 2 examples:
# 1) k-means on a random generated dataset
# 2) Using k-means for customer segmentation (Cust_Segmentation.csv) dataset

import random
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
import pandas as pd
from sklearn.preprocessing import StandardScaler
from mpl_toolkits.mplot3d import Axes3D

########### 1)
print('\n--> K-MEANS ON RAMDOM DATASET\n')

### RANDOM GENERATED DATASET
# Lets create our own dataset for this lab!
# First we need to set up a random seed. Use numpy's random.seed() function, where the seed will be set to 0
np.random.seed(0)

# Next we will be making random clusters of points by using the make_blobs class. The make_blobs class can take in many
# inputs, but we will be using these specific ones.
# Input
#  - n_samples: The total number of points equally divided among clusters. Value will be: 5000
#  - centers: The number of centers to generate, or the fixed center locations. Value will be: centers
#  - cluster_std: The standard deviation of the clusters. Value will be: 0.9
# Output
#  - X: Array of shape [n_samples, n_features]. (Feature Matrix) The generated samples.
#  - y: Array of shape [n_samples]. (Response Vector) The integer labels for cluster membership of each sample.
centers = [[4, 4], [-2, -1], [2, -3], [1, 1]]
X, y = make_blobs(n_samples=5000, centers=centers, cluster_std=0.9)

# Display the scatter plot of the randomly generated data.
plt.figure(figsize=(8, 6))
plt.scatter(X[:, 0], X[:, 1], marker='.')
plt.title('Randomly Generated Blobs')
plt.show()

### K-MEANS MODEL
# Now that we have our random data, let's set up our K-Means Clustering.
# The KMeans class has many parameters that can be used, but we will be using these three:
#  - init: Initialization method of the centroids. Value will be: "k-means++"
#  - k-means++: Selects initial cluster centers for k-mean clustering in a smart way to speed up convergence.
#  - n_clusters: The number of clusters to form as well as the number of centroids to generate. Value will be: 4
#  - n_init: Number of time the k-means algorithm will be run with different centroid seeds. The final results will be
#  the best output of n_init consecutive runs in terms of inertia. Value will be: 12
# Initialize KMeans with these parameters, where the output parameter is called k_means.
k_means = KMeans(init="k-means++", n_clusters=4, n_init=12)

# Now let's fit the KMeans model with the feature matrix we created above, X.
k_means.fit(X)

# Now let's grab the labels for each point in the model using KMeans' labels_ attribute and save it
k_means_labels = k_means.labels_

# We will also get the coordinates of the cluster centers using KMeans' cluster and centers and save it
k_means_cluster_centers = k_means.cluster_centers_

### PLOTTING
# Now that we have the random data generated and the KM model initialized, let's plot them and see what it looks like!
# Initialize the plot with the specified dimensions.
fig = plt.figure(figsize=(8, 6))
# Colors uses a color map, which will produce an array of colors based on the number of labels there are. We use set
# (k_means_labels) to get the unique labels.
colors = ['r', 'g', 'b', 'y']
# colors = plt.cm.Spectral(np.linspace(0, 1, len(set(k_means_labels))))
# Create a plot
ax = fig.add_subplot(1, 1, 1)
ax.set_title('K-means with 4 clusters')
# For loop that plots the data points and centroids. k will range from 0-3, which will match the possible clusters that
# each data point is in.
for k, col in zip(range(len([[4, 4], [-2, -1], [2, -3], [1, 1]])), colors):
    # Create a list of all data points, where the data points that are in the cluster (ex. cluster 0) are labeled as
    # true, else they are labeled as false.
    my_members = (k_means_labels == k)

    # Define the centroid, or cluster center.
    cluster_center = k_means_cluster_centers[k]

    # Plots the datapoints with color col.
    ax.plot(X[my_members, 0], X[my_members, 1], 'w', markerfacecolor=col, marker='.')

    # Plots the centroids with specified color, but with a darker outline
    ax.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col, markeredgecolor='k', markersize=6)
# Remove x-axis ticks
ax.set_xticks(())
# Remove y-axis ticks
ax.set_yticks(())
# Show the plot
plt.show()

### CHANGE K
# Try to cluster the above dataset into 3 clusters.
k_means3 = KMeans(init="k-means++", n_clusters=3, n_init=12)
k_means3.fit(X)
fig = plt.figure(figsize=(8, 6))
colors = ['r', 'g', 'b']
ax = fig.add_subplot(1, 1, 1)
ax.set_title('K-means with 3 clusters')
for k, col in zip(range(len(k_means3.cluster_centers_)), colors):
    my_members = (k_means3.labels_ == k)
    cluster_center = k_means3.cluster_centers_[k]
    ax.plot(X[my_members, 0], X[my_members, 1], 'w', markerfacecolor=col, marker='.')
    ax.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col, markeredgecolor='k', markersize=6)
plt.show()

########### 2)
print('\n\n--> K-MEANS ON IMPORTED DATASET')

### IMPORT AND EXPLORE DATASET
# Imagine that you have a customer dataset, and you need to apply customer segmentation on this historical data.
# Customer segmentation is the practice of partitioning a customer base into groups of individuals that have similar
# characteristics. It is a significant strategy as a business can target these specific groups of customers and
# effectively allocate marketing resources. For example, one group might contain customers who are high-profit and
# low-risk, that is, more likely to purchase products, or subscribe for a service. A business task is to retaining those
# customers. Another group might include customers from non-profit organizations. And so on.
# Import Cust_Segmentation.csv dataset, previously downloaded (at ./Data/) from the IBM Object Storage, from:
# https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/ML0101ENv3/labs/Cust_Segmentation.csv
df = pd.read_csv("./Data/Cust_Segmentation.csv")
# df = pd.read_csv("/home/simonet/PycharmProjects/PyLearn/7_ScikitLearn-Scipy/IBM_MLcourse/3_Clustering/Data/Cust_Segmentation.csv")

pd.set_option("display.max_columns", None, 'display.width', None)
print('\nData structure:')
print(df.head())  # take a look at the dataset

### SELECT AND NORMALIZE DATA
# As you can see, Address in this dataset is a categorical variable. k-means algorithm isn't directly applicable to
# categorical variables because Euclidean distance function isn't really meaningful for discrete variables. So, lets
# drop this feature and run clustering.
cdf = df.drop('Address', axis=1)
print('\nStructure of data with non-categorical features only:')
print(cdf.head())

# Now let's normalize the dataset over the standard deviation. But why do we need normalization in the first place?
# Normalization is a statistical method that helps mathematical-based algorithms to interpret features with different
# magnitudes and distributions equally. We use StandardScaler() to normalize our dataset.
X = cdf.values[:, 1:]
X = np.nan_to_num(X)
Clus_dataSet = StandardScaler().fit_transform(X)

### K-MEANS MODEL
# In our example (if we didn't have access to the k-means algorithm), it would be the same as guessing that each
# customer group would have certain age, income, education, etc, with multiple tests and experiments. However, using the
# K-means clustering we can do all this process much easier. Lets apply k-means and take a look at cluster labels.
clusterNum = 3
k_means = KMeans(init="k-means++", n_clusters=clusterNum, n_init=12)
k_means.fit(X)
labels = k_means.labels_

# We assign the labels to each row in dataframe.
cdf["Clus_km"] = labels
print('\nData with new column (Clus_km) representing the cluster assigned to each sample'
      'thanks to the K-Means method:')
print(cdf.head())

# We can easily check the centroid values by averaging the features in each cluster.
print('\nAll data samples grouped and averaged according to KM cluster of belonging (Clus_km):\n',
      cdf.groupby('Clus_km').mean())

### PLOTTING
# Now, lets look at the distribution of customers based on their Age and Income. The area of each dot in this
# scatterplot is given by customers' Education, and their color is given by the corresponding cluster assigned with the
# K-Means method.
area = np.pi * (X[:, 1]) ** 2
fig, ax = plt.subplots(figsize=(8, 6))
dots = ax.scatter(X[:, 0], X[:, 3], s=area, c=labels.astype(np.float), alpha=0.5)
ax.set_xlabel('Age', fontsize=18)
ax.set_ylabel('Income', fontsize=16)
legend = ax.legend(*dots.legend_elements(), title="Classes")
ax.add_artist(legend)
plt.show()

# Similarly, let's plot the 3 features above (Age, Education and Income) in a 3D plot and assign colors to the dots
# according to the corresponding clusters.
fig3d = plt.figure(1, figsize=(8, 6))
plt.clf()
ax = Axes3D(fig3d, rect=[0, 0, .95, 1], elev=48, azim=134)
plt.cla()
ax.set_xlabel('Education')
ax.set_ylabel('Age')
ax.set_zlabel('Income')
dots = ax.scatter(X[:, 1], X[:, 0], X[:, 3], c=labels.astype(np.float))
legend = ax.legend(*dots.legend_elements(), title="Classes")
ax.add_artist(legend)
plt.show()

# k-means will partition your customers into mutually exclusive groups, for example, into 3 clusters. The customers in
# each cluster are similar to each other demographically.
# Now we can create a profile for each group, considering the common characteristics of each cluster. 
# For example, the 3 clusters can be:
# - AFFLUENT, EDUCATED AND OLD AGED
# - MIDDLE AGED AND MIDDLE INCOME
# - YOUNG AND LOW INCOME

### NOTE: CHOOSING K (elbow method)
# The questions is, how can we evaluate the goodness of the clusters formed by k-Means? In other words, how do we
# calculate the accuracy of k-Means clustering? One way is to compare the clusters with the ground truth, if it's
# available. However, because k-Means is an unsupervised algorithm we usually don't have ground truth in real world
# problems to be used. But there is still a way to say how bad each cluster is, based on the objective of the k-Means.
# This value is the average distance between data points within a cluster. Also, average of the distances of data points
# from their cluster centroids can be used as a metric of error for the clustering algorithm. Essentially, determining
# the number of clusters in a data set, or k as in the k-Means algorithm, is a frequent problem in data clustering.
# The correct choice of K is often ambiguous because it's very dependent on the shape and scale of the distribution of
# points in a dataset. There are some approaches to address this problem, but one of the techniques that is commonly
# used is to run the clustering across the different values of K and looking at a metric of accuracy for clustering.
# This metric can be mean, distance between data points and their cluster's centroid, which indicate how dense our
# clusters are or, to what extent we minimize the error of clustering. Then, looking at the change of this metric, we
# can find the best value for K. But the problem is that with increasing the number of clusters, the distance of
# centroids to data points will always reduce. This means increasing K will always decrease the error. So, the value of
# the metric as a function of K is plotted and the elbow point is determined where the rate of decrease sharply shifts.
# It is the right K for clustering. This method is called the elbow method.
