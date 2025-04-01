import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split

class MovieLensDataset(Dataset):
    """PyTorch Dataset for MovieLens data."""
    def __init__(self, df):
        self.users = torch.tensor(df['userId'].values, dtype=torch.long)
        self.items = torch.tensor(df['movieId'].values, dtype=torch.long)
        self.labels = torch.tensor(df['label'].values, dtype=torch.float)

    def __len__(self):
        return len(self.users)

    def __getitem__(self, index):
        return self.users[index], self.items[index], self.labels[index]

def load_and_preprocess_data(ratings_file, num_negatives=4):
    """
    Loads MovieLens ratings data, converts explicit ratings to binary implicit feedback,
    and applies negative sampling efficiently.

    Args:
        ratings_file (str): Path to the ratings file.
        num_negatives (int): Number of negative samples per positive example.

    Returns:
        train_df, val_df, test_df: DataFrames for training, validation, and test splits.
        num_users (int): Total number of users.
        num_items (int): Total number of items.
    """
    # Load data
    data = pd.read_csv(ratings_file, sep='::', engine='python',
                       names=['userId', 'movieId', 'rating', 'timestamp'])

    # Convert explicit ratings to binary implicit feedback
    data['label'] = (data['rating'] >= 4).astype(int)

    # Map userId and movieId to contiguous indices
    unique_users = data['userId'].unique()
    unique_items = data['movieId'].unique()
    user2id = {user: idx for idx, user in enumerate(unique_users)}
    item2id = {item: idx for idx, item in enumerate(unique_items)}

    data['userId'] = data['userId'].map(user2id)
    data['movieId'] = data['movieId'].map(item2id)

    num_users = len(user2id)
    num_items = len(item2id)

    # Get positive interactions
    positive_samples = data[data['label'] == 1][['userId', 'movieId']]
    
    # Convert user-item interactions into a dictionary for fast lookup
    user_item_dict = data.groupby('userId')['movieId'].apply(set).to_dict()

    # Vectorized Negative Sampling
    users, items, labels = [], [], []

    all_items = np.array(list(range(num_items)))  # Faster item lookup

    for user, pos_item in zip(positive_samples['userId'], positive_samples['movieId']):
        users.append(user)
        items.append(pos_item)
        labels.append(1)  # Positive sample
        
        # Get negative samples efficiently
        neg_samples = np.random.choice(all_items, num_negatives * 2, replace=False)  # Sample more to ensure validity
        neg_samples = [neg for neg in neg_samples if neg not in user_item_dict[user]][:num_negatives]

        users.extend([user] * len(neg_samples))
        items.extend(neg_samples)
        labels.extend([0] * len(neg_samples))  # Negative samples

    # Create a DataFrame from arrays
    samples_df = pd.DataFrame({'userId': users, 'movieId': items, 'label': labels})

    # Stratified splitting
    train_df, temp_df = train_test_split(samples_df, test_size=0.3, random_state=42, stratify=samples_df['label'])
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42, stratify=temp_df['label'])

    return train_df, val_df, test_df, num_users, num_items
