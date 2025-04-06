import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

# -------------------------------
# Data Preprocessing and Dataset
# -------------------------------

class MovieLensDataset(Dataset):
    """
    PyTorch Dataset for MovieLens data.
    Expects a DataFrame with columns: ['userId', 'movieId', 'label'].
    """
    def __init__(self, df):
        self.df = df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        # Get a row and convert userId, movieId, label to tensors
        row = self.df.iloc[index]
        user = torch.tensor(row['userId'], dtype=torch.long)
        item = torch.tensor(row['movieId'], dtype=torch.long)
        label = torch.tensor(row['label'], dtype=torch.float)
        return user, item, label

def load_and_preprocess_data(ratings_file, num_negatives=4):
    """
    Loads MovieLens ratings data, converts explicit ratings to binary implicit feedback,
    and applies negative sampling.

    Args:
        ratings_file (str): Path to the ratings file (e.g., 'ratings.dat').
        num_negatives (int): Number of negative samples per positive example.
    
    Returns:
        train_df, val_df, test_df: DataFrames for training, validation, and test splits.
        num_users (int): Total number of users.
        num_items (int): Total number of items.
    """
    # Read the ratings file with separator '::' (using python engine due to multi-character separator)
    data = pd.read_csv(ratings_file, sep='::', engine='python',
                       names=['userId', 'movieId', 'rating', 'timestamp'])
    
    # Convert explicit ratings to binary feedback (1 for rating>=4, else 0)
    data['label'] = (data['rating'] >= 4).astype(int)
    
    # Create a mapping for user and item indices starting at 0
    # This step ensures our embedding layers have contiguous indices.
    unique_users = data['userId'].unique()
    unique_items = data['movieId'].unique()
    user2id = {user: idx for idx, user in enumerate(unique_users)}
    item2id = {item: idx for idx, item in enumerate(unique_items)}
    
    data['userId'] = data['userId'].map(user2id)
    data['movieId'] = data['movieId'].map(item2id)
    
    # Build a dictionary: user -> set of items they interacted with (for negative sampling)
    user_item_set = data.groupby('userId')['movieId'].apply(set).to_dict()
    all_items = set(data['movieId'].unique())
    
    # Negative sampling: For every positive interaction, sample 'num_negatives' negative interactions
    samples = []
    # Iterate only over positive interactions to ensure balance
    for idx, row in data[data['label'] == 1].iterrows():
        u = row['userId']
        pos_item = row['movieId']
        samples.append((u, pos_item, 1))
        # Randomly sample negative items not in the user's interaction set
        for _ in range(num_negatives):
            neg_item = np.random.choice(list(all_items - user_item_set[u]))
            samples.append((u, neg_item, 0))
    
    # Create a DataFrame from sampled interactions
    samples_df = pd.DataFrame(samples, columns=['userId', 'movieId', 'label'])
    
    # Split dataset into training (70%), validation (15%), and testing (15%)
    train_df, temp_df = train_test_split(samples_df, test_size=0.3, random_state=42, stratify=samples_df['label'])
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42, stratify=temp_df['label'])
    
    num_users = len(user2id)
    num_items = len(item2id)
    
    return train_df, val_df, test_df, num_users, num_items

# -------------------------------
# Neural Collaborative Filtering Model
# -------------------------------

class NCF(nn.Module):
    """
    Neural Collaborative Filtering (NCF) model combining GMF and MLP branches.
    """
    def __init__(self, num_users, num_items, embedding_dim=32, mlp_layers=[64, 32, 16, 8]):
        """
        Initializes the NCF model.

        Args:
            num_users (int): Total number of users.
            num_items (int): Total number of items.
            embedding_dim (int): Dimensionality for user/item embeddings.
            mlp_layers (list): List containing sizes of MLP hidden layers.
        """
        super(NCF, self).__init__()
        
        # Embedding layers for the GMF branch
        self.user_embedding_gmf = nn.Embedding(num_users, embedding_dim)
        self.item_embedding_gmf = nn.Embedding(num_items, embedding_dim)
        
        # Embedding layers for the MLP branch
        self.user_embedding_mlp = nn.Embedding(num_users, embedding_dim)
        self.item_embedding_mlp = nn.Embedding(num_items, embedding_dim)
        
        # Build MLP layers dynamically based on provided mlp_layers list.
        mlp_input_dim = embedding_dim * 2  # since we concatenate user and item embeddings
        mlp_modules = []
        for layer_size in mlp_layers:
            mlp_modules.append(nn.Linear(mlp_input_dim, layer_size))
            mlp_modules.append(nn.ReLU())
            mlp_input_dim = layer_size
        self.mlp = nn.Sequential(*mlp_modules)
        
        # Fusion layer: Concatenation of GMF (element-wise product) and MLP output.
        fusion_dim = embedding_dim + mlp_layers[-1]
        self.predict_layer = nn.Linear(fusion_dim, 1)
        self.sigmoid = nn.Sigmoid()  # Final activation for probability output

    def forward(self, user_indices, item_indices):
        """
        Forward pass to compute the predicted probability of interaction.

        Args:
            user_indices (Tensor): Tensor of user indices.
            item_indices (Tensor): Tensor of item indices.

        Returns:
            Tensor: Predicted probability (after sigmoid activation).
        """
        # GMF branch: element-wise multiplication of user and item embeddings.
        user_gmf = self.user_embedding_gmf(user_indices)
        item_gmf = self.item_embedding_gmf(item_indices)
        gmf_output = user_gmf * item_gmf
        
        # MLP branch: concatenation of user and item embeddings, then passing through MLP layers.
        user_mlp = self.user_embedding_mlp(user_indices)
        item_mlp = self.item_embedding_mlp(item_indices)
        mlp_input = torch.cat([user_mlp, item_mlp], dim=-1)
        mlp_output = self.mlp(mlp_input)
        
        # Fusion: concatenate outputs of GMF and MLP branches
        fusion_input = torch.cat([gmf_output, mlp_output], dim=-1)
        prediction = self.predict_layer(fusion_input)
        prediction = self.sigmoid(prediction)  # Convert to probability [0, 1]
        return prediction

# -------------------------------
# Training and Evaluation Functions
# -------------------------------

def train_model(model, train_loader, val_loader, num_epochs=50, learning_rate=0.001, patience=5):
    """
    Trains the NCF model using binary cross-entropy loss and Adam optimizer.
    Includes early stopping based on validation loss.

    Args:
        model (nn.Module): The NCF model.
        train_loader (DataLoader): DataLoader for training data.
        val_loader (DataLoader): DataLoader for validation data.
        num_epochs (int): Maximum number of epochs.
        learning_rate (float): Learning rate for Adam optimizer.
        patience (int): Number of epochs with no improvement after which training will be stopped.

    Returns:
        The trained model.
    """
    loss_function = nn.BCELoss()  # Binary cross-entropy loss for binary classification.
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    best_val_loss = float('inf')
    trigger_times = 0

    for epoch in range(num_epochs):
        model.train()  # Set model to training mode
        train_loss = 0.0
        for users, items, labels in train_loader:
            optimizer.zero_grad()  # Clear gradients
            predictions = model(users, items).squeeze()  # Forward pass
            loss = loss_function(predictions, labels)
            loss.backward()  # Backpropagation
            optimizer.step()  # Update weights
            train_loss += loss.item()
        
        # Compute average training loss for the epoch
        avg_train_loss = train_loss / len(train_loader)
        
        # Evaluate on the validation set
        model.eval()  # Set model to evaluation mode
        val_loss = 0.0
        with torch.no_grad():
            for users, items, labels in val_loader:
                predictions = model(users, items).squeeze()
                loss = loss_function(predictions, labels)
                val_loss += loss.item()
        avg_val_loss = val_loss / len(val_loader)
        
        print(f"Epoch {epoch+1}: Train Loss = {avg_train_loss:.4f}, Val Loss = {avg_val_loss:.4f}")
        
        # Early stopping check: if no improvement, increase trigger counter.
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            trigger_times = 0
        else:
            trigger_times += 1
            if trigger_times >= patience:
                print("Early stopping triggered.")
                break
                
    return model

def evaluate_model(model, test_df, train_user_items, all_items, top_k=10):
    """
    Evaluates the NCF model using Recall@10 and NDCG@10 metrics.

    Args:
        model (nn.Module): The trained NCF model.
        test_df (DataFrame): DataFrame with test interactions.
        train_user_items (dict): Dictionary mapping user to items seen in training.
        all_items (set): Set of all item indices.
        top_k (int): The number of top recommendations to consider.

    Returns:
        avg_recall, avg_ndcg: The average Recall@10 and NDCG@10 scores.
    """
    model.eval()  # Set model to evaluation mode
    recall_scores = []
    ndcg_scores = []
    
    # Get list of unique users in the test set
    unique_users = test_df['userId'].unique()
    
    # Evaluate for each user individually
    for user in unique_users:
        # Test items: items that the user interacted with in the test set (positive interactions)
        test_items = test_df[test_df['userId'] == user]['movieId'].tolist()
        
        # Candidate items: items not seen in training by the user
        candidate_items = list(all_items - train_user_items.get(user, set()))
        if len(candidate_items) == 0:
            continue

        # Create tensors for the user and candidate items
        user_tensor = torch.tensor([user]*len(candidate_items), dtype=torch.long)
        item_tensor = torch.tensor(candidate_items, dtype=torch.long)
        
        with torch.no_grad():
            predictions = model(user_tensor, item_tensor).squeeze().cpu().numpy()
        
        # Rank items based on prediction scores in descending order
        ranked_indices = np.argsort(predictions)[::-1]
        top_items = [candidate_items[i] for i in ranked_indices[:top_k]]
        
        # Compute Recall@10: Fraction of test items found in top_k recommendations.
        hit_set = set(top_items) & set(test_items)
        recall = len(hit_set) / len(test_items)
        recall_scores.append(recall)
        
        # Compute NDCG@10:
        dcg = sum((1/np.log2(rank+2)) for rank, item in enumerate(top_items) if item in test_items)
        ideal_dcg = sum((1/np.log2(i+2)) for i in range(min(len(test_items), top_k)))
        ndcg = dcg / ideal_dcg if ideal_dcg > 0 else 0
        ndcg_scores.append(ndcg)
    
    avg_recall = np.mean(recall_scores)
    avg_ndcg = np.mean(ndcg_scores)
    return avg_recall, avg_ndcg

# -------------------------------
# Main Execution
# -------------------------------

def main():
    # Set random seed for reproducibility
    np.random.seed(42)
    torch.manual_seed(42)
    
    # Load and preprocess the data. Adjust the path to your ratings file.
    ratings_file = 'ratings.dat'  # Replace with the correct path if needed.
    train_df, val_df, test_df, num_users, num_items = load_and_preprocess_data(ratings_file)
    
    # Create PyTorch datasets and loaders
    batch_size = 256
    train_dataset = MovieLensDataset(train_df)
    val_dataset = MovieLensDataset(val_df)
    test_dataset = MovieLensDataset(test_df)  # Although test_dataset is not used in evaluation here.
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Initialize the NCF model
    model = NCF(num_users, num_items, embedding_dim=32, mlp_layers=[64, 32, 16, 8])
    
    # Train the model with early stopping based on validation loss
    trained_model = train_model(model, train_loader, val_loader, num_epochs=50, learning_rate=0.001, patience=5)
    
    # Prepare dictionary of user -> items interacted with in the training set (for evaluation)
    train_user_items = train_df.groupby('userId')['movieId'].apply(set).to_dict()
    all_items = set(range(num_items))
    
    # Evaluate the trained model using Recall@10 and NDCG@10
    avg_recall, avg_ndcg = evaluate_model(trained_model, test_df, train_user_items, all_items, top_k=10)
    print(f"Test Recall@10: {avg_recall:.4f}")
    print(f"Test NDCG@10: {avg_ndcg:.4f}")

if __name__ == '__main__':
    main()
