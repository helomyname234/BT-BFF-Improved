"""
Siamese Network Module for Feature Dimensionality Reduction

Implements the Siamese network architecture as described in Section 3.1 of the paper.

The Siamese network:
- Maps input data to a projection space
- Uses Euclidean distance as similarity measure
- Uses Triplet Loss for training
- Consists of 3-layer MLP with shared parameters

Reference: Section 3.1 of Wang et al. (2024)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class TripletLoss(nn.Module):
    """
    Triplet Margin Loss for Siamese network training.
    
    Goal is to force d(anchor, positive) < d(anchor, negative) + margin
    
    Args:
        margin: The margin for dissimilar pairs (default: 0.3)
    """
    
    def __init__(self, margin: float = 0.3):
        super(TripletLoss, self).__init__()
        self.criterion = nn.TripletMarginLoss(margin=margin, p=2)
    
    def forward(
        self, 
        anchor: torch.Tensor, 
        positive: torch.Tensor, 
        negative: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute Triplet Margin Loss.
        
        Args:
            anchor: Anchor embedding tensor (batch_size, embedding_dim)
            positive: Positive embedding tensor (batch_size, embedding_dim)
            negative: Negative embedding tensor (batch_size, embedding_dim)
            
        Returns:
            Triplet loss value
        """
        return self.criterion(anchor, positive, negative)


class SiameseNetwork(nn.Module):
    """
    Siamese Network for feature dimensionality reduction.
    
    Architecture as per Section 3.1:
    - Three-layer MLP with shared parameters
    - Input layer -> Hidden layer (5 neurons) -> Output layer
    - Uses Tanh activation
    
    The network transforms input features to a lower-dimensional embedding space
    where samples of the same category are closer together.
    
    Args:
        input_dim: Dimension of input features (78 for CIC-IDS2017, 43 for TON_IoT)
        hidden_dim: Dimension of hidden layer (default: 5 as per paper)
        output_dim: Dimension of output embedding (default: 36 to reshape to 6x6)
    """
    
    def __init__(
        self, 
        input_dim: int, 
        hidden_dim: int = 5,
        output_dim: int = 36
    ):
        super(SiameseNetwork, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        # Shared MLP with 3 layers as per paper
        # Input layer -> Hidden layer -> Output layer
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),  # Tanh activation as per paper
            nn.Linear(hidden_dim, output_dim),
            nn.Tanh()
        )
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights using Xavier initialization"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward_one(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for a single input.
        
        Args:
            x: Input tensor (batch_size, input_dim)
            
        Returns:
            Embedding tensor (batch_size, output_dim)
        """
        return self.encoder(x)
    
    def forward(
        self, 
        anchor: torch.Tensor, 
        positive: torch.Tensor,
        negative: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass for a triplet of inputs.
        
        Args:
            anchor: Anchor input tensor (batch_size, input_dim)
            positive: Positive input tensor (batch_size, input_dim)
            negative: Negative input tensor (batch_size, input_dim)
            
        Returns:
            Tuple of (embedding_a, embedding_p, embedding_n)
        """
        embedding_a = self.forward_one(anchor)
        embedding_p = self.forward_one(positive)
        embedding_n = self.forward_one(negative)
        return embedding_a, embedding_p, embedding_n
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode input features to embedding space.
        
        This is used after training to transform the dataset
        for the Predecessor/Successor models.
        
        Args:
            x: Input tensor (batch_size, input_dim)
            
        Returns:
            Embedding tensor (batch_size, output_dim)
        """
        return self.forward_one(x)
    
    def encode_and_reshape(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode and reshape features to image format (6x6x1).
        
        As per Section 4.2: "a reshaping operation was performed to transform
        the network traffic into a 6x6x1 feature map"
        
        Args:
            x: Input tensor (batch_size, input_dim)
            
        Returns:
            Reshaped tensor (batch_size, 1, 6, 6)
        """
        embedding = self.encode(x)
        # Reshape to 6x6x1 (C=1, H=6, W=6)
        batch_size = embedding.shape[0]
        return embedding.view(batch_size, 1, 6, 6)
    
    @property
    def num_parameters(self) -> int:
        """Return total number of trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class SiameseTrainer:
    """
    Trainer class for Siamese Network.
    
    Handles:
    - Training with Triplet Loss
    - Feature encoding after training
    """
    
    def __init__(
        self,
        model: SiameseNetwork,
        margin: float = 1.0,
        learning_rate: float = 0.0001,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        self.model = model.to(device)
        self.device = device
        self.criterion = TripletLoss(margin=margin)
        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        
    def train_epoch(self, dataloader) -> float:
        """
        Train for one epoch.
        
        Args:
            dataloader: DataLoader with SiameseTripletDataset
            
        Returns:
            Average loss for the epoch
        """
        self.model.train()
        total_loss = 0.0
        
        for batch_idx, (anchor, positive, negative) in enumerate(dataloader):
            anchor = anchor.to(self.device)
            positive = positive.to(self.device)
            negative = negative.to(self.device)
            
            self.optimizer.zero_grad()
            
            # Forward pass
            emb_a, emb_p, emb_n = self.model(anchor, positive, negative)
            
            # Compute loss
            loss = self.criterion(emb_a, emb_p, emb_n)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
        
        return total_loss / len(dataloader)
    
    def encode_dataset(self, features: torch.Tensor, batch_size: int = 1024) -> torch.Tensor:
        """
        Encode entire dataset using trained Siamese network.
        
        Args:
            features: Input features tensor
            batch_size: Batch size for encoding
            
        Returns:
            Encoded features tensor
        """
        self.model.eval()
        encoded_features = []
        
        with torch.no_grad():
            for i in range(0, len(features), batch_size):
                batch = features[i:i+batch_size].to(self.device)
                encoded = self.model.encode(batch)
                encoded_features.append(encoded.cpu())
        
        return torch.cat(encoded_features, dim=0)
