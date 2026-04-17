"""
BT-TPF Training Pipeline

Complete training pipeline implementing the 5 steps from Section 3.5:
1. Data pre-processing
2. Dimensionality reduction using Siamese network
3. Pre-training Predecessor and Successor models
4. Knowledge distillation using improved BERT-of-Theseus
5. Testing the Successor model

Reference: Section 3.5 of Wang et al. (2024)
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
import numpy as np
from typing import Dict, Tuple, Optional
from tqdm import tqdm

from .config import BTPTFConfig
from .models.siamese_network import SiameseNetwork, TripletLoss, SiameseTrainer
from .models.predecessor import Predecessor
from .models.successor import Successor
from .models.bert_of_theseus import BERTOfTheseus
from .data.preprocessing import DataPreprocessor, SiameseTripletDataset, IntrusionDataset
from .utils.metrics import compute_metrics, print_metrics, print_model_info


class BTPTFTrainer:
    """
    Complete BT-TPF Training Pipeline.
    
    Implements the flow chart from Figure 6:
    1. Pre-processed data → Siamese network → Embedding
    2. Reshape embedding to 6×6×1 feature map
    3. Pre-train Predecessor and Successor
    4. BERT-of-Theseus knowledge distillation
    5. Fine-tune and test Successor
    
    Args:
        config: BTPTFConfig with all hyperparameters
    """
    
    def __init__(self, config: BTPTFConfig):
        self.config = config
        self.device = torch.device(
            config.device if torch.cuda.is_available() else 'cpu'
        )
        print(f"Using device: {self.device}")
        
        # Models (initialized during training)
        self.siamese_network = None
        self.predecessor = None
        self.successor = None
        self.bert_of_theseus = None
        
        # Data preprocessor
        self.preprocessor = DataPreprocessor()
        
        # Training history
        self.history = {}
    
    @staticmethod
    def _build_weighted_sampler(labels: np.ndarray, power: float = 1.0) -> WeightedRandomSampler:
        """
        Build a WeightedRandomSampler with inverse-frequency class weights.
        
        Args:
            labels: Class index labels for the training subset
            power: Weight exponent. 1.0 means strict inverse-frequency.
            
        Returns:
            WeightedRandomSampler for balanced mini-batches over epochs
        """
        class_counts = np.bincount(labels)
        class_counts = np.maximum(class_counts, 1)
        class_weights = 1.0 / np.power(class_counts, power)
        sample_weights = class_weights[labels]
        sample_weights = torch.from_numpy(sample_weights).float()
        return WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True
        )
    
    def prepare_data(
        self,
        train_features: np.ndarray,
        train_labels: np.ndarray,
        test_features: np.ndarray,
        test_labels: np.ndarray
    ) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """
        Prepare data loaders for training.
        
        Args:
            train_features: Training features
            train_labels: Training labels
            test_features: Test features
            test_labels: Test labels
            
        Returns:
            Tuple of (train_loader, val_loader, test_loader)
        """
        # Create Siamese triplet dataset for Siamese training
        siamese_dataset = SiameseTripletDataset(
            train_features, 
            train_labels,
            num_triplets=len(train_features)
        )
        
        siamese_loader = DataLoader(
            siamese_dataset,
            batch_size=self.config.training.batch_size,
            shuffle=True,
            num_workers=0
        )
        
        return siamese_loader, train_features, train_labels, test_features, test_labels
    
    def train_siamese_network(
        self,
        train_features: np.ndarray,
        train_labels: np.ndarray,
        epochs: int = 50
    ) -> SiameseNetwork:
        """
        Step 2: Train Siamese network for dimensionality reduction.
        
        As per Section 3.1:
        - 3-layer MLP with shared parameters
        - Triplet Loss with margin=0.3
        - Output: 36-dimensional embedding
        
        Args:
            train_features: Training features
            train_labels: Training labels
            epochs: Number of training epochs
            
        Returns:
            Trained Siamese network
        """
        print("\n" + "="*60)
        print("Step 2: Training Siamese Network for Dimensionality Reduction")
        print("="*60)
        
        input_dim = train_features.shape[1]
        
        # Initialize Siamese network
        self.siamese_network = SiameseNetwork(
            input_dim=input_dim,
            hidden_dim=self.config.siamese.hidden_dim,
            output_dim=self.config.siamese.output_dim
        )
        
        print_model_info(self.siamese_network, "Siamese Network")
        
        # Create trainer
        trainer = SiameseTrainer(
            model=self.siamese_network,
            margin=self.config.siamese.margin,
            learning_rate=self.config.training.learning_rate,
            device=self.device
        )
        
        # Create dataset
        dataset = SiameseTripletDataset(
            train_features,
            train_labels,
            balance_by_class=self.config.data.siamese_balance_by_class
        )
        dataloader = DataLoader(
            dataset,
            batch_size=self.config.training.batch_size,
            shuffle=True
        )
        
        # Train
        losses = []
        for epoch in range(epochs):
            loss = trainer.train_epoch(dataloader)
            losses.append(loss)
            
            if (epoch + 1) % 10 == 0:
                print(f"  Epoch {epoch+1}/{epochs}, Loss: {loss:.4f}")
        
        self.history['siamese_loss'] = losses
        
        return self.siamese_network
    
    def encode_features(
        self,
        features: np.ndarray,
        batch_size: int = 1024
    ) -> torch.Tensor:
        """
        Encode features using trained Siamese network.
        
        Reshapes to 6×6×1 feature map as per Section 4.2:
        "a reshaping operation was performed to transform the 
        network traffic into a 6x6x1 feature map"
        
        Args:
            features: Input features
            batch_size: Batch size for encoding
            
        Returns:
            Encoded and reshaped features (N, 1, 6, 6)
        """
        self.siamese_network.eval()
        encoded_features = []
        
        features_tensor = torch.tensor(features, dtype=torch.float32)
        
        with torch.no_grad():
            for i in range(0, len(features), batch_size):
                batch = features_tensor[i:i+batch_size].to(self.device)
                encoded = self.siamese_network.encode_and_reshape(batch)
                encoded_features.append(encoded.cpu())
        
        return torch.cat(encoded_features, dim=0)
    
    def train_full_pipeline(
        self,
        train_features: np.ndarray,
        train_labels: np.ndarray,
        test_features: np.ndarray,
        test_labels: np.ndarray,
        num_classes: int
    ) -> Dict:
        """
        Execute the complete BT-TPF training pipeline.
        
        Steps from Section 3.5:
        1. Data pre-processing (done externally)
        2. Dimensionality reduction using Siamese network
        3. Pre-train Predecessor and Successor
        4. Knowledge distillation (BERT-of-Theseus)
        5. Fine-tune and test Successor
        
        Args:
            train_features: Training features (preprocessed)
            train_labels: Training labels
            test_features: Test features (preprocessed)
            test_labels: Test labels
            num_classes: Number of classes
            
        Returns:
            Dictionary with training history and results
        """
        # Step 2: Train Siamese Network
        self.train_siamese_network(
            train_features, 
            train_labels,
            epochs=self.config.training.pretrain_epochs
        )
        
        # Encode features
        print("\nEncoding features with Siamese network...")
        train_encoded = self.encode_features(train_features)
        test_encoded = self.encode_features(test_features)
        
        # Create datasets
        train_dataset = IntrusionDataset(
            train_encoded.numpy(), 
            train_labels
        )
        test_dataset = IntrusionDataset(
            test_encoded.numpy(), 
            test_labels
        )
        
        # Split training for validation
        train_size = int(0.8 * len(train_dataset))
        val_size = len(train_dataset) - train_size
        train_subset, val_subset = torch.utils.data.random_split(
            train_dataset, [train_size, val_size]
        )
        
        if self.config.training.use_weighted_sampler:
            # Use subset labels (NOT full train labels) to avoid leakage/misalignment.
            train_subset_indices = np.array(train_subset.indices)
            train_subset_labels = train_labels[train_subset_indices]
            sampler = self._build_weighted_sampler(
                labels=train_subset_labels,
                power=self.config.training.weighted_sampler_power
            )
            train_loader = DataLoader(
                train_subset,
                batch_size=self.config.training.batch_size,
                sampler=sampler
            )
        else:
            train_loader = DataLoader(
                train_subset,
                batch_size=self.config.training.batch_size,
                shuffle=True
            )
        val_loader = DataLoader(
            val_subset,
            batch_size=self.config.training.batch_size,
            shuffle=False
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.config.training.batch_size,
            shuffle=False
        )
        
        # Step 3 & 4: Initialize and train with BERT-of-Theseus
        print("\n" + "="*60)
        print("Steps 3-4: BERT-of-Theseus Knowledge Distillation")
        print("="*60)
        
        # Initialize models
        self.predecessor = Predecessor(
            input_channels=self.config.predecessor.input_channels,
            input_size=self.config.predecessor.input_size,
            embed_dim=self.config.predecessor.embed_dim,
            num_modules=self.config.predecessor.num_modules,
            blocks_per_module=self.config.predecessor.blocks_per_module,
            num_heads=self.config.predecessor.num_heads,
            mlp_ratio=self.config.predecessor.mlp_ratio,
            num_classes=num_classes,
            dropout=self.config.predecessor.dropout
        )
        
        self.successor = Successor(
            input_channels=self.config.successor.input_channels,
            input_size=self.config.successor.input_size,
            embed_dim=self.config.successor.embed_dim,
            num_modules=self.config.successor.num_modules,
            blocks_per_module=self.config.successor.blocks_per_module,
            mlp_hidden_dim=self.config.successor.mlp_hidden_dim,  # 1 neuron as per paper
            num_classes=num_classes,
            dropout=self.config.successor.dropout
        )
        
        print_model_info(self.predecessor, "Predecessor (Teacher)")
        print_model_info(self.successor, "Successor (Student)")
        
        # Initialize BERT-of-Theseus
        self.bert_of_theseus = BERTOfTheseus(
            predecessor=self.predecessor,
            successor=self.successor,
            device=str(self.device),
            initial_replacement_rate=self.config.training.initial_replacement_rate,
            use_optimization=self.config.training.use_gradient_optimization
        )
        
        # Full training pipeline
        kd_history = self.bert_of_theseus.full_training_pipeline(
            train_loader=train_loader,
            val_loader=val_loader,
            pre_train_epochs=self.config.training.pretrain_epochs,
            replacement_epochs=self.config.training.replacement_epochs,
            fine_tune_epochs=self.config.training.finetune_epochs,
            learning_rate=self.config.training.learning_rate
        )
        
        self.history.update(kd_history)
        
        # Step 5: Test the Successor model
        print("\n" + "="*60)
        print("Step 5: Testing Successor Model")
        print("="*60)
        
        results = self.evaluate(test_loader)
        
        return {
            'history': self.history,
            'results': results
        }
    
    def evaluate(self, test_loader: DataLoader) -> Dict:
        """
        Evaluate the trained Successor model.
        
        Args:
            test_loader: DataLoader for test data
            
        Returns:
            Dictionary with evaluation metrics
        """
        self.successor.eval()
        
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for data, target in test_loader:
                data = data.to(self.device)
                output = self.successor(data)
                preds = output.argmax(dim=1)
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(target.numpy())
        
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        
        # Compute metrics
        metrics = compute_metrics(all_labels, all_preds)
        print_metrics(metrics, "BT-TPF (Successor)")
        
        return {
            'metrics': metrics,
            'predictions': all_preds,
            'labels': all_labels
        }
    
    def evaluate_predecessor(self, test_loader: DataLoader) -> Dict:
        """
        Evaluate the Predecessor model (baseline comparison).
        
        Args:
            test_loader: DataLoader for test data
            
        Returns:
            Dictionary with evaluation metrics
        """
        self.predecessor.eval()
        
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for data, target in test_loader:
                data = data.to(self.device)
                output = self.predecessor(data)
                preds = output.argmax(dim=1)
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(target.numpy())
        
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        
        metrics = compute_metrics(all_labels, all_preds)
        print_metrics(metrics, "Predecessor (Teacher)")
        
        return {
            'metrics': metrics,
            'predictions': all_preds,
            'labels': all_labels
        }
    
    def save_models(self, path: str):
        """Save trained models"""
        torch.save({
            'siamese_network': self.siamese_network.state_dict(),
            'predecessor': self.predecessor.state_dict(),
            'successor': self.successor.state_dict()
        }, path)
        print(f"Models saved to {path}")
    
    def load_models(self, path: str):
        """Load trained models"""
        checkpoint = torch.load(path, map_location=self.device)
        self.siamese_network.load_state_dict(checkpoint['siamese_network'])
        self.predecessor.load_state_dict(checkpoint['predecessor'])
        self.successor.load_state_dict(checkpoint['successor'])
        print(f"Models loaded from {path}")
