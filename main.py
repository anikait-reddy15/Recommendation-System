import pandas as pd
import numpy as np
import os
from recommender_engine import ContentRecommender # Import your class

# --- DATA LOADING SECTION ---
dataset_path = r"C:\Projects\Recommendation-System\Dataset"
movies_path = os.path.join(dataset_path, "movies.csv")
ratings_path = os.path.join(dataset_path, "ratings.csv")

dataset_path = r"C:\Projects\Recommendation-System\Dataset" # Dataset folder path

movies_path = os.path.join(dataset_path, "movies.csv") # Movies dataset path
ratings_path = os.path.join(dataset_path, "ratings.csv") # Ratings dataset path

try:
    # Movies DataFrame
    movies_df = pd.read_csv(movies_path, low_memory=False, usecols=["movieId", "title", "genres"],
                            dtype={"movieId" : "int32", "title" : "string", "genres" : "string"}) 
    # Ratings DataFrame
    ratings_df = pd.read_csv(ratings_path, low_memory=False, usecols=["userId", "movieId", "rating"],
                            dtype={"userId" : "int32", "movieId" : "int32", "rating" : "float32"}) 
    
except Exception as e:
    print(f"Exception : {e}")

# 1. Filter out movies with low number of ratings
movie_counts = ratings_df.groupby("movieId").size()
popular_movies = movie_counts[movie_counts >= 20].index
ratings_df = ratings_df[ratings_df["movieId"].isin(popular_movies)]

# This removes the "Ghost Movies" that have no ratings left
movies_df = movies_df[movies_df["movieId"].isin(popular_movies)]

# 2. Filter out users with low number of ratings
user_counts = ratings_df.groupby("userId").size()
active_users = user_counts[user_counts >= 50].index
ratings_df = ratings_df[ratings_df["userId"].isin(active_users)]

# Reset index to avoid issues later
movies_df = movies_df.reset_index(drop=True)

# 3. Clean Genres and Year
# Extract year (Visual only for now)
movies_df['year'] = movies_df['title'].str.extract('(\(\d{4}\))', expand=False)
movies_df['year'] = movies_df['year'].str.extract('(\d{4})', expand=False)
movies_df['title'] = movies_df['title'].str.replace('(\(\d{4}\))', '', regex=True).str.strip()

# Split genres for the visual dataframe
movies_df['genres'] = movies_df['genres'].str.split('|')

# 4. Create the Feature Matrix (Clean Math Only)
# We flatten the genres back to strings just for get_dummies, then join
dummies = movies_df['genres'].apply(lambda x: '|'.join(x) if isinstance(x, list) else x).str.get_dummies()
# genres_df should ONLY have movieId and the dummy columns
genres_df = pd.concat([movies_df['movieId'], dummies], axis=1)

# --- 1. NORMALIZATION (Mean-Centering) ---
# Calculate the mean rating given by each user
user_means = ratings_df.groupby('userId')['rating'].transform('mean')
# Subtract the mean to center ratings around zero
ratings_df['normalized_rating'] = ratings_df['rating'] - user_means

# --- 2. WEIGHTED RATING CALCULATION (IMDb Style) ---
# v = number of ratings for the movie
# m = minimum ratings required to be listed (we use your threshold of 20)
# R = average rating of the movie
# C = the mean rating across the whole report
v = ratings_df.groupby('movieId')['rating'].count()
R = ratings_df.groupby('movieId')['rating'].mean()
C = ratings_df['rating'].mean()
m = 20

weighted_rating = (v / (v + m) * R) + (m / (v + m) * C)
# Add this back to movies_df to see which movies are statistically "best"
movies_df = movies_df.merge(weighted_rating.rename('weighted_score'), on='movieId', how='left')

# --- 3. SPARSITY ANALYSIS (EDA) ---
n_users = ratings_df['userId'].nunique()
n_movies = ratings_df['movieId'].nunique()
n_ratings = len(ratings_df)

sparsity = (1 - n_ratings / (n_users * n_movies)) * 100
print(f"Dataset Sparsity: {sparsity:.2f}%")
# High sparsity (e.g., >99%) explains why we use Sparse Matrices later.

# --- 4. TOP 10 MOVIES BY WEIGHTED SCORE (EDA) ---
top_movies = movies_df.sort_values('weighted_score', ascending=False).head(10)
print("\nTop 10 Movies by Weighted Rating:")
print(top_movies[['title', 'weighted_score']])

# --- ENGINE SECTION ---
# Now this will work because the variables exist!
engine = ContentRecommender(movies_df, genres_df)

# Compute the similarity
engine.compute_similarity()

# Get recommendations
results = engine.get_recommendations('Toy Story')
print(results)