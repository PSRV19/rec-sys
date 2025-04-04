import torch
import numpy as np
import matplotlib.pyplot as plt

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

def plot_learning_curves(train_losses, val_losses, config, save_path=None):
    """
    Plot the training and validation loss curves.
    
    Args:
        train_losses (list): List of training losses per epoch.
        val_losses (list): List of validation losses per epoch.
        save_path (str, optional): Path to save the plot. If None, the plot is displayed.
    """
    plt.figure(figsize=(10, 6))
    epochs = range(1, len(train_losses) + 1)
    
    plt.plot(epochs, train_losses, 'b-', label='Training Loss')
    plt.plot(epochs, val_losses, 'r-', label='Validation Loss')
    
    plt.title('Training and Validation Loss for the NCF Model with the following configuration:\n' +
              f"Embedding Dim: {config['embedding_dim']}, MLP Layers: {config['mlp_layers']}, Dropout: {config['dropout']}, " +
              f"Learning Rate: {config['learning_rate']}, Weight Decay: {config['weight_decay']}, " +
              f"Batch Size: {config['batch_size']}, Num Negatives: {config['num_negatives']}")
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()
    
    plt.close()

def plot_metrics_comparison(model_configs, metrics, metric_name, config, save_path=None):
    """
    Plot comparison of metrics across different model configurations.
    
    Args:
        model_configs (list): List of model configuration names.
        metrics (list): List of metric values corresponding to each model configuration.
        metric_name (str): Name of the metric being compared (e.g., 'Recall@10', 'NDCG@10').
        save_path (str, optional): Path to save the plot. If None, the plot is displayed.
    """
    plt.figure(figsize=(10, 6))
    
    # Create bar chart
    plt.bar(model_configs, metrics)
    
    plt.title(f'{metric_name} Comparison Across Model Configurations for the following configuration\n' +
              f"Embedding Dim: {config['embedding_dim']}, MLP Layers: {config['mlp_layers']}, Dropout: {config['dropout']}, " +
              f"Learning Rate: {config['learning_rate']}, Weight Decay: {config['weight_decay']}, " +
              f"Batch Size: {config['batch_size']}, Num Negatives: {config['num_negatives']}")
    plt.xlabel('Model Configuration')
    plt.ylabel(metric_name)
    plt.xticks(rotation=45)
    plt.grid(True, axis='y')
    
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()
    
    plt.close()

def evaluate_at_different_k(model, test_df, train_user_items, all_items, k_values=[5, 10, 15, 20]):
    """
    Evaluate the model at different k values for top-k recommendations.
    
    Args:
        model (nn.Module): The trained NCF model.
        test_df (DataFrame): DataFrame with test interactions.
        train_user_items (dict): Dictionary mapping user to items seen in training.
        all_items (set): Set of all item indices.
        k_values (list): List of k values to evaluate.
        
    Returns:
        recall_at_k (dict): Dictionary mapping k values to Recall@k.
        ndcg_at_k (dict): Dictionary mapping k values to NDCG@k.
    """
    recall_at_k = {}
    ndcg_at_k = {}
    
    for k in k_values:
        recall, ndcg = evaluate_model(model, test_df, train_user_items, all_items, top_k=k)
        recall_at_k[k] = recall
        ndcg_at_k[k] = ndcg
    
    return recall_at_k, ndcg_at_k

def plot_metrics_at_k(k_values, recall_at_k, ndcg_at_k, save_path=None):
    """
    Plot Recall@k and NDCG@k for different values of k.
    
    Args:
        k_values (list): List of k values.
        recall_at_k (dict): Dictionary mapping k values to Recall@k.
        ndcg_at_k (dict): Dictionary mapping k values to NDCG@k.
        save_path (str, optional): Path to save the plot. If None, the plot is displayed.
    """
    plt.figure(figsize=(12, 6))
    
    # Create subplot for Recall@k
    plt.subplot(1, 2, 1)
    plt.plot(k_values, [recall_at_k[k] for k in k_values], 'bo-')
    plt.title('Recall@k')
    plt.xlabel('k')
    plt.ylabel('Recall')
    plt.grid(True)
    
    # Create subplot for NDCG@k
    plt.subplot(1, 2, 2)
    plt.plot(k_values, [ndcg_at_k[k] for k in k_values], 'ro-')
    plt.title('NDCG@k')
    plt.xlabel('k')
    plt.ylabel('NDCG')
    plt.grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()
    
    plt.close()