import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
import os

# Import from our modules
from data_preprocessing import load_and_preprocess_data, MovieLensDataset
from ncf_model import NCF
from train import train_model
from evaluate import (
    evaluate_model, 
    plot_learning_curves, 
    evaluate_at_different_k, 
    plot_metrics_at_k
)

def load_config(config_path):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def main():
    # Load configuration
    print("Loading configuration from config.yaml file...")
    config = load_config('config.yaml')
    
    # Set random seed for reproducibility
    print("Setting random seed for reproducibility...")
    random_seed = config['random_seed']
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    
    print("Creating output directory results/ for plots..")
    # Create output directory for plots if it doesn't exist
    output_dir = "results"
    os.makedirs(output_dir, exist_ok=True)
    
    print("Loading and preprocessing data...")
    # Load and preprocess the data. 
    ratings_file = 'ml-1m/ratings.dat'  
    train_df, val_df, test_df, num_users, num_items = load_and_preprocess_data(
        ratings_file, 
        num_negatives=config['num_negatives']
    )
    
    print("Creating PyTorch datasets and loaders...")
    # Create PyTorch datasets and loaders
    batch_size = config['batch_size']
    train_dataset = MovieLensDataset(train_df)
    val_dataset = MovieLensDataset(val_df)
    test_dataset = MovieLensDataset(test_df)  # Although test_dataset is not used in evaluation here.
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    print("Initializing the NCF model...")
    # Initialize the NCF model
    model = NCF(
        num_users, 
        num_items, 
        embedding_dim=config['embedding_dim'],
        mlp_layers=config['mlp_layers'],
        use_gmf=config['use_gmf'],
        use_mlp=config['use_mlp'],
        activation_function=config['activation_function']
    )
    
    print("Training the model (with early stopping)...")
    # Train the model with early stopping based on validation loss
    trained_model, train_losses, val_losses = train_model(
        model, 
        train_loader, 
        val_loader, 
        num_epochs=config['num_epochs'], 
        learning_rate=config['learning_rate'],
        weight_decay=config['weight_decay'],
        patience=config['early_stopping_patience']
    )
    
    # Plot learning curves
    plot_learning_curves(
        train_losses, 
        val_losses, 
        save_path=os.path.join(output_dir, "learning_curves.png")
    )
    
    # Prepare dictionary of user -> items interacted with in the training set (for evaluation)
    train_user_items = train_df.groupby('userId')['movieId'].apply(set).to_dict()
    all_items = set(range(num_items))
    
    # Evaluate the trained model using Recall@10 and NDCG@10
    avg_recall, avg_ndcg = evaluate_model(trained_model, test_df, train_user_items, all_items, top_k=10)
    print(f"Test Recall@10: {avg_recall:.4f}")
    print(f"Test NDCG@10: {avg_ndcg:.4f}")
    
    # Evaluate at different k values
    k_values = [5, 10, 15, 20, 50]
    recall_at_k, ndcg_at_k = evaluate_at_different_k(
        trained_model, 
        test_df, 
        train_user_items, 
        all_items, 
        k_values=k_values
    )
    
    # Plot metrics at different k values
    plot_metrics_at_k(
        k_values, 
        recall_at_k, 
        ndcg_at_k, 
        save_path=os.path.join(output_dir, "metrics_at_k.png")
    )
    
    # Print metrics at different k values
    print("\nMetrics at different k values:")
    for k in k_values:
        print(f"k={k}: Recall@{k}={recall_at_k[k]:.4f}, NDCG@{k}={ndcg_at_k[k]:.4f}")

if __name__ == '__main__':
    main()