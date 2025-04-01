import torch
import torch.nn as nn

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