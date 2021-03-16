#!/usr/bin/env python
# coding: utf-8

# Recommendation systems are a collection of algorithms used to recommend items to users based on information taken from
# the user. These systems have become ubiquitous, and can be commonly seen in online stores, movies databases and job
# finders. In this notebook, we will explore Content-based recommendation systems and implement a simple version of one
# using Python and the Pandas library.

import pandas as pd
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

# Let's look at the result!
print('\nData structure with new column ("year"):')
print(movies_df.head())

# With that, let's also split the values in the Genres column into a list of Genres to simplify future use. This can be
# achieved by applying Python's split string function on the correct column. Since every genre is separated by a | we
# simply have to call the split function on |
movies_df['genres'] = movies_df.genres.str.split('|')
# Since keeping genres in a list format isn't optimal for the content-based recommendation system technique, we will use
# the One Hot Encoding technique to convert the list of genres to a vector where each column corresponds to one possible
# value of the feature. This encoding is needed for feeding categorical data. In this case, we store every different
# genre in columns that contain either 1 or 0. 1 shows that a movie has that genre and 0 shows that it doesn't. Let's
# also store this dataframe in another variable since genres won't be important for our first recommendation system.

# Copying the movie dataframe into a new one since we won't need to use the genre information in our first case.
moviesWithGenres_df = movies_df.copy()

# For every row in the dataframe, iterate through the list of genres and place a 1 into the corresponding column
for index, row in movies_df.iterrows():
    for genre in row['genres']:
        moviesWithGenres_df.at[index, genre] = 1
# Filling in the NaN values with 0 to show that a movie doesn't have that column's genre
moviesWithGenres_df = moviesWithGenres_df.fillna(0)
moviesWithGenres_df.head()

# - RATINGS data structure
print('\nRatings data structure:')
print(ratings_df.head())

# Every row in the ratings dataframe has a user id associated with at least one movie, a rating and a timestamp showing
# when they reviewed it. We won't be needing the timestamp column, so let's drop it to save on memory.
# Drop removes a specified row or column from a dataframe
ratings_df = ratings_df.drop('timestamp', 1)
print('\nRatings data structure without "timestamp" column:')
print(ratings_df.head())

### CONTENT-BASED RECOMMENDATION SYSTEM
# Now, let's take a look at how to implement Content-Based or Item-Item recommendation systems. This technique attempts
# to figure out what a user's favourite aspects of an item is, and then recommends items that present those aspects.
# In our case, we're going to try to figure out the input's favorite genres from the movies and ratings given.

# ### Create list of movies and corresponding ratings of the customer we want to reccommend movies to
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

# With the input complete, let's extract the input movie's ID's from the movies dataframe and add them into it.
# We can achieve this by first filtering out the rows that contain the input movie's title and then merging this subset
# with the input dataframe. We also drop unnecessary columns for the input to save memory space.

# Filtering out the movies by title: assign to each title in inputMovies an ID corresponding to the location of that
# title in the movies_df dataframe
inputId = movies_df[movies_df['title'].isin(inputMovies['title'].tolist())]
# Then merging it so we can get the movieId. It's implicitly merging it by title.
inputMovies = pd.merge(inputId, inputMovies)
# Dropping information we won't use from the input dataframe
inputMovies = inputMovies.drop('genres', 1).drop('year', 1)
# If a movie you added in above isn't here, then it might not be in the original dataframe or it might be spelled
# differently, please check capitalisation.
print("\nAdding movieId column to the list of movies rated by the customer in exam:")
print(inputMovies.head())

# We're going to start by learning the input's preferences, so let's get the subset of movies that the input has watched
# from the Dataframe containing genres defined with binary values.

# Filtering out the movies from the input
userMovies = moviesWithGenres_df[moviesWithGenres_df['movieId'].isin(inputMovies['movieId'].tolist())]
print("\nAdding genres to the list of movies rated by the customer in exam:")
print(userMovies.head())

# We'll only need the actual genre table, so let's clean this up a bit by resetting the index and dropping the movieId,
# title, genres and year columns.

# Resetting the index to avoid future issues
userMovies = userMovies.reset_index(drop=True)
# Dropping unnecessary issues due to save memory and to avoid issues
userGenreTable = userMovies.drop('movieId', 1).drop('title', 1).drop('genres', 1).drop('year', 1)
print("\nOnly genres of the list of movies rated by the customer in exam:")
print(userGenreTable.head())

# Now we're ready to start learning the input's preferences!
# To do this, we're going to turn each genre into weights. We can do this by using the input's reviews and multiplying
# them into the input's genre table and then summing up the resulting table by column. This operation is actually a dot
# product between a matrix and a vector, so we can simply accomplish by calling Pandas's "dot" function.
userProfile = userGenreTable.transpose().dot(inputMovies['rating'])
# The user profile
print("\nWeights (from 0 to 10) assigned to each genre according to the preference of the customer in exam:")
print(userProfile)

# Now, we have the weights for every of the user's preferences. This is known as the User Profile. Using this, we can
# recommend movies that satisfy the user's preferences.

# Let's start by extracting the genre table from the original dataframe:
genreTable = moviesWithGenres_df.set_index(moviesWithGenres_df['movieId'])
# And drop the unnecessary information
genreTable = genreTable.drop('title', 1).drop('genres', 1).drop('year', 1).drop('movieId', 1)
print("\nGenres of movies in the whole list of movies:")
print(genreTable.head())

# With the input's profile and the complete list of movies and their genres in hand, we're going to take the weighted
# average of every movie based on the input profile and recommend the top twenty movies that most satisfy it.

# Creates an empty dataframe
recommendation_df = pd.DataFrame()
# recommendation_df['movieId'] = genreTable.index
# Multiply the genres by the weights and then take the weighted average
recommendationTable = ((genreTable * userProfile).sum(axis=1)) / (userProfile.sum())
recommendation_df['recommendation score'] = recommendationTable
# Add title column
recommendation_df = recommendation_df.merge(movies_df, left_on='movieId', right_on='movieId', how='inner')
recommendation_df = recommendation_df.drop('year', 1)
# Sort our recommendations in descending order
recommendation_df = recommendation_df.sort_values(by='recommendation score', ascending=False)
# Just a peek at the values
print("\nPercentage value for each movie of recommendation score to the customer in exam:")
print(recommendation_df.head(10))
# # Now here's the recommendation table!
# movies_df.loc[movies_df['movieId'].isin(recommendationTable.head(20).keys())]


# ### Advantages and Disadvantages of Content-Based Filtering
# 
# ##### Advantages
# * Learns user's preferences
# * Highly personalized for the user
# 
# ##### Disadvantages
# * Doesn't take into account what others think of the item, so low quality item recommendations might happen
# * Extracting data is not always intuitive
# * Determining what characteristics of the item the user dislikes or likes is not always obvious
