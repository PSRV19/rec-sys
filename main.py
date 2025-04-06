import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
import os
import datetime
import csv

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

config = {
    'embedding_dim': 32,  # Keep the embedding dimension as is
    'mlp_layers': [64, 32, 16, 8],  # Keep the MLP architecture as is
    'dropout': 0.2,  # Set dropout to 0.2 for all layers

    # Training settings
    'num_epochs': 20,  # Run for 20 epochs
    'batch_size': 128,  
    'learning_rate': 0.0005,
    'early_stopping_patience': 3,
    'step_size': 5,
    'gamma': 0.25,
    'weight_decay': 1e-5,  # Add weight decay to the optimizer

    # Dataset settings
    'num_negatives': 8,  # Keep the number of negative samples as is
    'train_split': 0.7,
    'val_split': 0.15,
    'test_split': 0.15,

    # Other
    'random_seed': 42
}

def main():
    
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
        dropout=config['dropout']
    )
    
    print("Training the model (with early stopping)...")
    # Train the model with early stopping based on validation loss
    trained_model, train_losses, val_losses = train_model(
        model, 
        train_loader, 
        val_loader, 
        num_epochs=config['num_epochs'], 
        learning_rate=config['learning_rate'],
        patience=config['early_stopping_patience'],
        step_size=config['step_size'],
        gamma=config['gamma']
    )
    
    # Plot learning curves
    plot_learning_curves(
        train_losses, 
        val_losses,
        config,
        save_path=os.path.join(output_dir, f"learning_curves_{datetime.datetime.now().strftime("%d_%m_%Y__%H_%M")}.png")
    )

    # Get final training and validation loss
    final_train_loss = train_losses[-1]
    final_val_loss = val_losses[-1]
    
    # Prepare dictionary of user -> items interacted with in the training set (for evaluation)
    train_user_items = train_df.groupby('userId')['movieId'].apply(set).to_dict()
    all_items = set(range(num_items))
    
    # Evaluate the trained model using Recall@10 and NDCG@10
    avg_recall, avg_ndcg = evaluate_model(trained_model, test_df, train_user_items, all_items, top_k=10)
    print(f"Test Recall@10: {avg_recall:.4f}")
    print(f"Test NDCG@10: {avg_ndcg:.4f}")

    # Save metrics to a file
    results_file = os.path.join(output_dir, f"results_{datetime.datetime.now().strftime("%d_%m_%Y__%H_%M")}.csv")
    file_exists = os.path.isfile(results_file)
    
    with open(results_file, mode='a', newline='') as file:
        writer = csv.writer(file)
        # Write header if the file is new
        if not file_exists:
            writer.writerow([
                "embedding_dim", "mlp_layers", "dropout", "batch_size", 
                "num_epochs", "learning_rate", "final_train_loss", 
                "final_val_loss", "recall@10", "ndcg@10"
            ])
        # Write metrics and configuration
        writer.writerow([
            config['embedding_dim'], config['mlp_layers'], config['dropout'], config['batch_size'], 
            config['num_epochs'], config['learning_rate'], final_train_loss, 
            final_val_loss, avg_recall, avg_ndcg
        ])
    
    print("Metrics saved to results.csv")
    
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
        config,
        save_path=os.path.join(output_dir, f"metrics_at_k_{datetime.datetime.now().strftime("%d_%m_%Y__%H_%M")}.png")
    )
    
    # Print metrics at different k values
    print("\nMetrics at different k values:")
    for k in k_values:
        print(f"k={k}: Recall@{k}={recall_at_k[k]:.4f}, NDCG@{k}={ndcg_at_k[k]:.4f}")
    print("\nEvaluation complete.")
    print("=" * 50 + "\n")

if __name__ == '__main__':
    # run default configuration
    main()

    # Uncomment the following lines to run with different configurations
    # Configuration 1
    config['embedding_dim'] = 64
    main()

    # Configuration 2
    config['embedding_dim'] = 64
    config['batch_size'] = 256
    main()

    # Configuration 3
    config['embedding_dim'] = 32
    config['batch_size'] = 256
    main()

    # Configuration 4
    config['embedding_dim'] = 32
    config['mlp_layers'] = [128, 64, 32, 16]
    config['dropout'] = 0.2
    config['batch_size'] = 128
    main()

    # Configuration 5
    config['embedding_dim'] = 64
    main()

    # Configuration 6
    config['embedding_dim'] = 32
    config['batch_size'] = 256
    main()

    # Configuration 7
    config['embedding_dim'] = 64
    main()

    # Configuration 8
    config['num_negatives'] = 4
    config['embedding_dim'] = 32
    config['mlp_layers'] = [64, 32, 16, 8]
    config['batch_size'] = 128
    main()

    # Configuration 9
    config['num_negatives'] = 10
    main()

    
    
