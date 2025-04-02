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

    data['userId'] = data['userId'].map(lambda x: user2id[x])
    data['movieId'] = data['movieId'].map(lambda x: item2id[x])

    num_users = len(user2id)
    num_items = len(item2id)

    # Get positive interactions
    positive_samples = data[data['label'] == 1][['userId', 'movieId']]
    
    # Convert user-item interactions into a dictionary for fast lookup
    user_item_dict = {user: set(items) for user, items in data.groupby('userId')['movieId']}

    # Precompute negative candidates per user
    negative_candidates = {}
    all_items = np.arange(num_items)  # Faster item lookup

    for user in range(num_users):
        negative_candidates[user] = np.setdiff1d(all_items, list(user_item_dict.get(user, set())), assume_unique=True)

    # Generate negative samples in a vectorized manner
    users = np.repeat(positive_samples['userId'].values, num_negatives + 1)
    items = np.empty_like(users)
    labels = np.empty_like(users)

    # Assign positive samples
    items[::num_negatives + 1] = positive_samples['movieId'].values
    labels[::num_negatives + 1] = 1

    # Assign negative samples
    for i, user in enumerate(positive_samples['userId'].values):
        neg_samples = np.random.choice(negative_candidates[user], num_negatives, replace=False)
        start_idx = i * (num_negatives + 1) + 1
        items[start_idx:start_idx + num_negatives] = neg_samples
        labels[start_idx:start_idx + num_negatives] = 0

    # Create DataFrame from NumPy arrays
    samples_df = pd.DataFrame({'userId': users, 'movieId': items, 'label': labels})

    # Stratified splitting
    train_df, temp_df = train_test_split(samples_df, test_size=0.3, random_state=42, stratify=samples_df['label'])
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42, stratify=temp_df['label'])

    return train_df, val_df, test_df, num_users, num_items
