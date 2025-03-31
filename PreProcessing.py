import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

class PreProcessing:
    def __init__(self, data):
        self.data = data
    
    # Load in the users dataset (Don't really need it I think)
    def get_users(self):
        self.users_format = ['user_id', 'gender', 'age', 'occupation', 'zip']
        self.users = pd.read_table('ml-1m/users.dat', sep='::', header=None, names=self.users_format, engine='python')
        print("Users Shape: ", self.users.shape)
        
        return self.users
    
    # Load in the ratings dataset (Relevant columns are user_id, movie_id, and rating)
    def get_ratings(self):
        self.ratings_format = ['user_id', 'movie_id', 'rating', 'timestamp']
        self.ratings = pd.read_table('ml-1m/ratings.dat', sep='::', header=None, names=self.ratings_format, engine='python')
        print("Ratings Shape: ", self.ratings.shape)
        
        return self.ratings
    
    # Load in the movies dataset (Will be useful in the end to get the movie titles)
    def get_movies(self):
        self.movies_format = ['movie_id', 'title', 'genres']
        self.movies = pd.read_table('ml-1m/movies.dat', sep='::', header=None, names=self.movies_format, engine='python')
        print("Movies Shape: ", self.movies.shape)
        
        return self.movies
    
    # Convert user ids into an embedding to use for input into the model
    def encode_users(self):
        self.user_encoder = LabelEncoder()
        self.ratings['user_id'] = self.user_encoder.fit_transform(self.ratings['user_id'])
        
        return self.ratings
    
    # Convert movie ids into an embedding to use for input into the model
    def encode_movies(self):
        self.movie_encoder = LabelEncoder()
        self.ratings['movie_id'] = self.movie_encoder.fit_transform(self.ratings['movie_id'])
        
        return self.ratings
    
    # Convert ratings into binary implicit feedback
    def binary_convert(self):
        self.ratings['rating'] = self.ratings['rating'].apply(lambda x: 1 if x >= 3 else 0)
        
        return self.ratings 
    
    # Implement negative sampling 
    def negative_sampling(self):
        """
        Instead of assuming that every unrated item is a negative example 
        (which is inefficient), we randomly sample a few negative interactions 
        (items the user hasn’t interacted with) and label them as 0 (disliked).
        """
        # Get all unique user-item pairs
        self.user_item_pairs = pd.MultiIndex.from_prduct(
            [self.ratings['user_id'].unique(), 
             self.ratings['movie_id'].unique()], 
            names=['user_id', 'movie_id']
        )
        
        # Get the user-item pairs that *are* rated
        self.rated_pairs = self.ratings.set_index(['user_id','movie_id']).index
        
        # Determine the negative samples by finding the difference
        self.negative_samples = self.user_item_pairs.difference(self.rated_pairs)
        
        # Randomly sample a fraction of the negative samples
        self.negative_samples_df = pd.DataFrame(self.negative_samples, columns=['user_id','movie_id'])
        self.negative_samples_df['rating'] = 0
        
        # Combine with the original ratings
        self.ratings = pd.concat([self.ratings, self.negative_samples_df]).reset_index(drop=True)

        return self.ratings
        
        