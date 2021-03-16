#!/usr/bin/env python
# coding: utf-8

# Recommendation systems are a collection of algorithms used to recommend items to users based on information taken from
# the user. These systems have become ubiquitous, and can be commonly seen in online stores, movies databases and job
# finders. In this notebook, we will explore Collaborative Filtering recommendation systems and implement a simple
# version of one using Python and the Pandas library.

import pandas as pd
from math import sqrt
import matplotlib.pyplot as plt

### IMPORT AND EXPLORE DATASET
# Import moviedataset.csv dataset, previously downloaded and unziped (at ./Data/) from the IBM Object Storage:
# https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/ML0101ENv3/labs/moviedataset.zip
# Note: Dataset was acquired from [GroupLens] (http://grouplens.org/datasets/movielens/).
# Storing the movie information into a pandas dataframe
# movies_df = pd.read_csv('./Data/movies.csv')
movies_df = pd.read_csv(
    '/home/simonet/PycharmProjects/PyLearn/7_ScikitLearn-Scipy/IBM_MLcourse/4_Recommender Systems/Data/movies.csv')
# Storing the user information into a pandas dataframe
# ratings_df = pd.read_csv('./Data/ratings.csv')
ratings_df = pd.read_csv(
    '/home/simonet/PycharmProjects/PyLearn/7_ScikitLearn-Scipy/IBM_MLcourse/4_Recommender Systems/Data/ratings.csv')

# - MOVIES data structure
pd.set_option("display.max_columns", None, 'display.width', None)
print('\nMovie data structure:')
print(movies_df.head())  # take a look at the the first 5 rows of a dataframe

# So each movie has a unique ID, a title with its release year along with it (Which may contain unicode characters) and
# several different genres in the same field. Let's remove the year from the title column and place it into its own one
# by using the handy pandas.Series.str.extract function that Pandas has.

# Let's remove the year from the title column by using pandas' replace function and store in a new year column.
# Using regular expressions to find a year stored between parentheses.
# We specify the parentheses so we don't conflict with movies that have years in their titles
movies_df['year'] = movies_df.title.str.extract('(\(\d\d\d\d\))', expand=False)
# Removing the parentheses
movies_df['year'] = movies_df.year.str.extract('(\d\d\d\d)', expand=False)
# Removing the years from the 'title' column
movies_df['title'] = movies_df.title.str.replace('(\(\d\d\d\d\))', '')
# Applying the strip function to get rid of any ending whitespace characters that may have appeared
movies_df['title'] = movies_df['title'].apply(lambda x: x.strip())

# With that, let's also drop the genres column since we won't need it for this particular recommendation system.
movies_df = movies_df.drop('genres', 1)

# Here's the final movies dataframe:
print('\nMovie data structure with new "year" column and without "genre" column:')
print(movies_df.head())

# - RATINGS data structure
print('\nRatings data structure:')
print(ratings_df.head())

# Every row in the ratings dataframe has a user id associated with at least one movie, a rating and a timestamp showing
# when they reviewed it. We won't be needing the timestamp column, so let's drop it to save on memory.
# Drop removes a specified row or column from a dataframe
ratings_df = ratings_df.drop('timestamp', 1)

# Here's how the final ratings Dataframe looks like:
print('\nRatings data structure without "timestamp" column:')
print(ratings_df.head())

### COLLABORATIVE FILTERING RECOMMENDATION SYSTEM
# The technique we're going to take a look at is called Collaborative Filtering, which is also known as User-User
# Filtering. As hinted by its alternate name, this technique uses other users to recommend items to the input user. It
# attempts to find users that have similar preferences and opinions as the input and then recommends items that they
# have liked to the input. There are several methods of finding similar users (Even some making use of Machine
# Learning), and the one we will be using here is going to be based on the Pearson Correlation Function.

# The process for creating a User Based recommendation system is as follows:
# - Select a user with the movies the user has watched
# - Based on his rating to movies, find the top X neighbours 
# - Get the watched movie record of the user for each neighbour.
# - Calculate a similarity score using some formula
# - Recommend the items with the highest score

# #### Create list of movies and corresponding ratings of the customer we want to recommend movies to
# Let's begin by creating an input user to recommend movies to:
# Notice: To add more movies, simply increase the amount of elements in the userInput. Feel free to add more in!
# Just be sure to write it in with capital letters and if a movie starts with a "The", like "The Matrix" then write it
# in like this: 'Matrix, The'.
userInput = [
    {'title': 'Breakfast Club, The', 'rating': 5},
    {'title': 'Toy Story', 'rating': 3.5},
    {'title': 'Jumanji', 'rating': 2},
    {'title': "Pulp Fiction", 'rating': 5},
    {'title': 'Akira', 'rating': 4.5}
]
inputMovies = pd.DataFrame(userInput)
print("\nRatings of the customer we are considering:")
print(inputMovies.head())

# With the input complete, let's extract the input movies's ID's from the movies dataframe and add them into it.
# We can achieve this by first filtering out the rows that contain the input movies' title and then merging this subset
# with the input dataframe. We also drop unnecessary columns for the input to save memory space.

# Filtering out the movies by title: assign to each title in inputMovies an ID corresponding to the location of that
# title in the movies_df dataframe
inputId = movies_df[movies_df['title'].isin(inputMovies['title'].tolist())]
# Then merging it so we can get the movieId. It's implicitly merging it by title.
inputMovies = pd.merge(inputId, inputMovies)
# Dropping information we won't use from the input dataframe
inputMovies = inputMovies.drop('year', 1)
# If a movie you added in above isn't here, then it might not be in the original dataframe or it might be spelled
# differently, please check capitalisation.
print("\nAdding movieId column to the list of movies rated by the customer in exam:")
print(inputMovies.head())

# #### Users who have seen the same movies
# Now with the movie ID's in our input, we can get the subset of users that have watched and reviewed the movies
# in our input.

# Filtering out users that have watched movies that the input has watched and storing it
userSubset = ratings_df[ratings_df['movieId'].isin(inputMovies['movieId'].tolist())]
print("\nUsers who have seen the same movies:")
print(userSubset.head())

# We now group up the rows by user ID.
# Groupby creates several sub dataframes where they all have the same value in the column specified as the parameter
userSubsetGroup = userSubset.groupby(['userId'])

# lets look at one of the users, e.g. the one with userID=1130
print("\nLets look, as an example, at the user with ID=1130:")
print(userSubsetGroup.get_group(1130))

# Let's also sort these groups so the users that share the most movies in common with the input have higher priority.
# This provides a richer recommendation since we won't go through every single user.

# Sorting it so users with movie most in common with the input will have priority
userSubsetGroup = sorted(userSubsetGroup, key=lambda x: len(x[1]), reverse=True)

# Now lets look at the first user
print("\nAfter sorting users for sharing the most movies in common with the customer in exam, "
      "lets look at the first one:")
print(userSubsetGroup[0][1])

# #### Similarity of users to input user
# Next, we are going to compare all users (not really all !!!) to our specified user and find the one that is most
# similar. we're going to find out how similar each user is to the input through the Pearson Correlation Coefficient.
# It is used to measure the strength of a linear association between two variables. The values given by the formula vary
# from r = -1 to r = 1, where 1 forms a direct correlation between the two entities (it means a perfect positive
# correlation) and -1 forms a perfect negative correlation. In our case, a 1 means that the two users have similar
# tastes while a -1 means the opposite.
# Why Pearson Correlation? It is invariant to scaling, i.e. multiplying all elements by a nonzero constant or adding any
# constant to all elements. For example, if you have two vectors X and Y, then, pearson(X, Y) == pearson(X, 2 * Y + 3).
# This is a pretty important property in recommendation systems because for example two users might rate two series of
# items totally different in terms of absolute rates, but they would be similar users (i.e. with similar ideas) with
# similar rates in various scales.

# We will select a subset of users to iterate through. This limit is imposed because we don't want to waste too much
# time going through every single user.
userSubsetGroup = userSubsetGroup[0:100]

# Now, we calculate the Pearson Correlation between input user and subset group, and store it in a dictionary, where the
# key is the user Id and the value is the coefficient.
pearsonCorrelationDict = {}
# For every user group in our subset
for name, group in userSubsetGroup:
    # Let's start by sorting the input and current user group so the values aren't mixed up later on
    group = group.sort_values(by='movieId')
    nRatings = len(group)  # Get the N for the formula
    inputMovies = inputMovies.sort_values(by='movieId')
    # Get the review scores for the movies that they both have in common
    temp_df = inputMovies[inputMovies['movieId'].isin(group['movieId'].tolist())]
    # And then store them in a temporary buffer variable in a list format to facilitate future calculations
    tempRatingList = temp_df['rating'].tolist()
    # Let's also put the current user group reviews in a list format
    tempGroupList = group['rating'].tolist()
    # Now let's calculate the pearson correlation between two users, so called, x and y
    Sxx = sum([i ** 2 for i in tempRatingList]) - pow(sum(tempRatingList), 2) / float(nRatings)
    Syy = sum([i ** 2 for i in tempGroupList]) - pow(sum(tempGroupList), 2) / float(nRatings)
    Sxy = sum(i * j for i, j in zip(tempRatingList, tempGroupList)) - sum(tempRatingList) * sum(tempGroupList) / float(
        nRatings)

    # If the denominator is different than zero, then divide, else, 0 correlation.
    if Sxx != 0 and Syy != 0:
        pearsonCorrelationDict[name] = Sxy / sqrt(Sxx * Syy)
    else:
        pearsonCorrelationDict[name] = 0

pearsonCorrelationDict.items()
pearsonDF = pd.DataFrame.from_dict(pearsonCorrelationDict, orient='index')
pearsonDF.columns = ['similarityIndex']
pearsonDF['userId'] = pearsonDF.index
pearsonDF.index = range(len(pearsonDF))
print("\nPerson Correlation (similarity in ratings) between user X and customer in exam:")
print(pearsonDF.head())

# Now let's get the top 50 users that are most similar to the input.
topUsers = pearsonDF.sort_values(by='similarityIndex', ascending=False)[0:50]
print("\nMost similar users:")
print(topUsers.head())

# #### Recommend movies
# We're going to do this by taking the weighted average of the ratings of the movies using the Pearson Correlation as
# the weight.
# But to do this, we first need to get the movies watched by the users in our pearsonDF from the ratings
# dataframe and then store their correlation in a new column called similarityIndex". This is achieved below by merging
# of these two tables. Lets take the rating of selected users to all movies:
topUsersRating = topUsers.merge(ratings_df, left_on='userId', right_on='userId', how='inner')
print("\nArray of movies and corresponding ratings of users sorted by similarity with customer in exam:")
print(topUsersRating.head())

# Now all we need to do is simply multiply the movie rating by its weight (The similarity index), then sum up the new
# ratings and divide it by the sum of the weights.
# We can easily do this by simply multiplying two columns, then grouping up the dataframe by movieId and then dividing
# two columns.

# Multiplies the similarity by the user's ratings
topUsersRating['weightedRating'] = topUsersRating['similarityIndex'] * topUsersRating['rating']
print(
    "\nAdd weightedRating column representing the rating of users weighted by their similarity with customer in exam:")
print(topUsersRating.head())

# Applies a sum to the topUsers after grouping it up by userId
tempTopUsersRating = topUsersRating.groupby('movieId').sum()[['similarityIndex', 'weightedRating']]
tempTopUsersRating.columns = ['sum_similarityIndex', 'sum_weightedRating']
print("\nArray of movies, sorted by their ID, with two columns for corresponding sum of weighted ratings and sum of "
      "similarity indices of users who rated them:")
print(tempTopUsersRating.head())

# Creates an empty dataframe
recommendation_df = pd.DataFrame()
# Now we take the weighted average
recommendation_df['recommendation score'] = \
    tempTopUsersRating['sum_weightedRating'] / tempTopUsersRating['sum_similarityIndex']
# Add title column
recommendation_df = recommendation_df.merge(movies_df, left_on='movieId', right_on='movieId', how='inner')
print("\nWeighted average recommendation score for each movie:")
print(recommendation_df.head())

# Now let's sort it and see the top 5 movies that the algorithm recommended!
recommendation_df = recommendation_df.sort_values(by='recommendation score', ascending=False)
print("\nRecommended movies sorted by their recommendation score:")
print(recommendation_df.head())

# Apply a threshold to recommendation score and recommend only movies with score higher than 4
th = 4
recommendation_th = recommendation_df[recommendation_df['recommendation score'] > th]
# Remove seen movies
seenMovies = []
for i in range(len(inputMovies)):
    if inputMovies.loc[i]['rating'] > th:
        seenMovies.append((recommendation_th['movieId'] == inputMovies.loc[i]['movieId']).idxmax())
recommendation_th = recommendation_th.drop(seenMovies)
recommendation_th = recommendation_th.drop('year', 1)
print("\nThe number of movies with recommendation score above {}/5 are {}.".format(th, len(recommendation_th)))
print("First 10 of those are:")
print(recommendation_th.head(10))


# ### Advantages and Disadvantages of Collaborative Filtering
#
# ##### Advantages
# * Takes other user's ratings into consideration
# * Doesn't need to study or extract information from the recommended item
# * Adapts to the user's interests which might change over time
# 
# ##### Disadvantages
# * Approximation function can be slow
# * There might be a low of amount of users to approximate
# * Privacy issues when trying to learn the user's preferences
