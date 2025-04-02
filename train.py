import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

def train_model(model, train_loader, val_loader, num_epochs=50, learning_rate=0.001, patience=10, weight_decay=0.01):
    """
    Trains the NCF model using binary cross-entropy loss and Adam optimizer.
    Includes early stopping based on validation loss and regularization to prevent overfitting.

    Args:
        model (nn.Module): The NCF model.
        train_loader (DataLoader): DataLoader for training data.
        val_loader (DataLoader): DataLoader for validation data.
        num_epochs (int): Maximum number of epochs.
        learning_rate (float): Learning rate for Adam optimizer.
        patience (int): Number of epochs with no improvement after which training will be stopped.
        weight_decay (float): L2 regularization strength to prevent overfitting.

    Returns:
        The trained model, training losses, and validation losses.
    """
    loss_function = nn.BCELoss()  # Binary cross-entropy loss for binary classification.
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
    # Learning rate scheduler to reduce learning rate when validation loss plateaus
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, verbose=True)
    
    best_val_loss = float('inf')
    trigger_times = 0
    best_model_state = None
    
    # Lists to store losses for each epoch
    train_losses = []
    val_losses = []

    for epoch in range(num_epochs):
        model.train()  # Set model to training mode
        train_loss = 0.0
        
        # Add progress bar for training batches
        train_progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]", 
                                  leave=False, ncols=100)
        
        for users, items, labels in train_progress_bar:
            optimizer.zero_grad()  # Clear gradients
            predictions = model(users, items).squeeze()  # Forward pass
            loss = loss_function(predictions, labels)
            loss.backward()  # Backpropagation
            
            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()  # Update weights
            train_loss += loss.item()
            
            # Update progress bar with current loss
            train_progress_bar.set_postfix(loss=f"{loss.item():.4f}")
        
        # Compute average training loss for the epoch
        avg_train_loss = train_loss / len(train_loader)
        train_losses.append(avg_train_loss)  # Save training loss
        
        # Evaluate on the validation set
        model.eval()  # Set model to evaluation mode
        val_loss = 0.0
        
        # Add progress bar for validation batches
        val_progress_bar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]", 
                                leave=False, ncols=100)
        
        with torch.no_grad():
            for users, items, labels in val_progress_bar:
                predictions = model(users, items).squeeze()
                loss = loss_function(predictions, labels)
                val_loss += loss.item()
                
                # Update progress bar with current loss
                val_progress_bar.set_postfix(loss=f"{loss.item():.4f}")
                
        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)  # Save validation loss
        
        # Update learning rate based on validation loss
        scheduler.step(avg_val_loss)
        
        print(f"Epoch {epoch+1}/{num_epochs}: Train Loss = {avg_train_loss:.4f}, Val Loss = {avg_val_loss:.4f}")
        
        # Early stopping check: if no improvement, increase trigger counter.
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            trigger_times = 0
            # Save the best model state
            best_model_state = model.state_dict().copy()
            print(f"New best model found with validation loss: {best_val_loss:.4f}")
        else:
            trigger_times += 1
            if trigger_times >= patience:
                # Load the best model state before returning
                print("Early stopping triggered.")
                if best_model_state is not None:
                    model.load_state_dict(best_model_state)
                    print(f"Loaded best model with validation loss: {best_val_loss:.4f}")
                break
    
    # Return the trained model along with the loss histories
    return model, train_losses, val_losses