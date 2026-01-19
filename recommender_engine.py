import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import os

class ContentRecommender:
    def __init__(self, movies_df, genres_df):
        self.movies_df = movies_df
        self.genres_df = genres_df
        self.sim_matrix = None

    def compute_similarity(self):
        # Prepare features (dropping movieId to keep only genre columns)
        features = self.genres_df.drop('movieId', axis=1)
        print("Computing Cosine Similarity Matrix...")
        self.sim_matrix = cosine_similarity(features, features)
        return self.sim_matrix

    def save_model(self, path='similarity_model.pkl'):
        with open(path, 'wb') as f:
            pickle.dump(self.sim_matrix, f)
        print(f"Model saved to {path}")

    def load_model(self, path='similarity_model.pkl'):
        if os.path.exists(path):
            with open(path, 'rb') as f:
                self.sim_matrix = pickle.load(f)
            print("Model loaded from disk.")
        else:
            print("Model file not found. Please compute similarity first.")

    def get_recommendations(self, movie_title, top_n=10):
        if self.sim_matrix is None:
            return "Error: Similarity matrix not loaded."

        # Find the index of the movie
        try:
            idx = self.movies_df.index[self.movies_df['title'] == movie_title][0]
        except IndexError:
            return "Movie not found in database."

        # Calculate scores
        sim_scores = list(enumerate(self.sim_matrix[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        
        # Get top results (excluding the movie itself)
        sim_scores = sim_scores[1:top_n+1]
        movie_indices = [i[0] for i in sim_scores]
        
        return self.movies_df[['title', 'genres']].iloc[movie_indices]