import pandas as pd
import numpy as np
import os

dataset_path = r"C:\Projects\Recommendation-System\Dataset" # Dataset folder path

movies_path = os.path.join(dataset_path, "movies.csv") # Movies dataset path
ratings_path = os.path.join(dataset_path, "ratings.csv") # Ratings dataset path

try:
    # Movies DataFrame
    movies_df = pd.read_csv(movies_path, low_memory=False, usecols=["movieId", "title", "genres"],
                            dtype={"movieId" : "int32", "title" : "string", "genres" : "string"}) 
    # Ratings DataFrame
    ratings_df = pd.read_csv(ratings_path, low_memory=False, usecols=["userId", "movieId", "rating"],
                            dtype={"userId" : "int32", "movieId" : "int32", "rating" : "float16"}) 
    
except Exception as e:
    print(f"Exception : {e}")