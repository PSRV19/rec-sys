from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim

def train_model(model, train_loader, val_loader, num_epochs=20, learning_rate=0.0005, patience=3, step_size=5, gamma=0.25, weight_decay=1e-5):
    """
    Trains the NCF model using binary cross-entropy loss and Adam optimizer.
    Includes a learning rate scheduler but disables early stopping.

    Args:
        model (nn.Module): The NCF model.
        train_loader (DataLoader): DataLoader for training data.
        val_loader (DataLoader): DataLoader for validation data.
        num_epochs (int): Number of epochs to train.
        learning_rate (float): Learning rate for Adam optimizer.
        step_size (int): Number of epochs before reducing the learning rate.
        gamma (float): Factor by which the learning rate is reduced.
        weight_decay (float): Weight decay for regularization.

    Returns:
        The trained model, training losses, and validation losses.
    """
    loss_function = nn.BCELoss()  # Binary cross-entropy loss
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)  # Add weight decay
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

    best_val_loss = float('inf')
    trigger_times = 0  # Counter for early stopping
    
    # Lists to store losses for each epoch
    train_losses = []
    val_losses = []

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0

        # Training loop
        train_progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]", leave=False, ncols=100)
        for users, items, labels in train_progress_bar:
            optimizer.zero_grad()
            
            # Get predictions and calculate training loss
            predictions = model(users, items).squeeze()
            loss = loss_function(predictions, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            train_progress_bar.set_postfix(loss=f"{loss.item():.4f}")

        avg_train_loss = train_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # Validation loop
        model.eval()
        val_loss = 0.0
        val_progress_bar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]", leave=False, ncols=100)
        with torch.no_grad():
            for users, items, labels in val_progress_bar:
                # Get predictions and calculate validation loss
                predictions = model(users, items).squeeze()
                loss = loss_function(predictions, labels)
                val_loss += loss.item()
                val_progress_bar.set_postfix(loss=f"{loss.item():.4f}")

        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)

        print(f"Epoch {epoch+1}/{num_epochs}: Train Loss = {avg_train_loss:.4f}, Val Loss = {avg_val_loss:.4f}")
        scheduler.step()

        # Early stopping check: if no improvement, increase trigger counter.
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            trigger_times = 0
        else:
            trigger_times += 1
            if trigger_times >= patience:
                print("Early stopping triggered.")
                break

    return model, train_losses, val_losses