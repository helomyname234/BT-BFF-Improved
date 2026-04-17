"""
Data Preprocessing Module for BT-TPF Framework

Implements:
- Label Encoding for character features
- Z-Score Normalization (Equation 20 from paper)
- Dataset classes for Siamese network training and classification

Reference: Section 4.2 of Wang et al. (2024)
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import LabelEncoder, StandardScaler
from typing import Tuple, Optional, List
import random


class DataPreprocessor:
    """
    Data preprocessor for intrusion detection datasets.
    
    Performs:
    1. Removal of NaN, Infinity, and null values
    2. Label Encoding for categorical features
    3. Z-Score Normalization (Equation 20):
        x_new = (x - μ) / σ
    """
    
    def __init__(self):
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.feature_columns = None
        self.target_column = None
        
    def fit_transform(self, df: pd.DataFrame, target_column: str = 'Label') -> Tuple[np.ndarray, np.ndarray]:
        """
        Fit the preprocessor and transform the data.
        
        Args:
            df: Input DataFrame
            target_column: Name of the target column
            
        Returns:
            Tuple of (features, labels)
        """
        self.target_column = target_column
        df = df.copy()
        
        # Step 1: Remove NaN, Infinity, and null values
        df = self._clean_data(df)
        
        # Separate features and target
        if target_column in df.columns:
            y = df[target_column].values
            X = df.drop(columns=[target_column])
        else:
            raise ValueError(f"Target column '{target_column}' not found in DataFrame")
        
        self.feature_columns = X.columns.tolist()
        
        # Step 2: Label Encoding for categorical features
        X = self._encode_categorical(X, fit=True)
        
        # Step 3: Z-Score Normalization (Equation 20)
        # x_new = (x - μ) / σ
        X = self.scaler.fit_transform(X)
        
        # Encode target labels
        y = self.label_encoder.fit_transform(y)
        
        return X.astype(np.float32), y.astype(np.int64)
    
    def transform(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Transform data using fitted preprocessor.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Tuple of (features, labels)
        """
        df = df.copy()
        
        # Clean data
        df = self._clean_data(df)
        
        # Separate features and target
        y = df[self.target_column].values
        X = df.drop(columns=[self.target_column])
        
        # Ensure columns match
        X = X[self.feature_columns]
        
        # Label Encoding
        X = self._encode_categorical(X, fit=False)
        
        # Z-Score Normalization
        X = self.scaler.transform(X)
        
        # Encode target labels
        y = self.label_encoder.transform(y)
        
        return X.astype(np.float32), y.astype(np.int64)
    
    def _clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Remove NaN, Infinity, and null values.
        Replace with 0 as per the paper.
        """
        # Replace infinity values with NaN first
        df = df.replace([np.inf, -np.inf], np.nan)
        
        # Replace NaN with 0
        df = df.fillna(0)
        
        # Replace dash '-' with 0 if present
        df = df.replace('-', 0)
        
        # Handle boolean features (True -> 1, False -> 0)
        for col in df.columns:
            if df[col].dtype == bool:
                df[col] = df[col].astype(int)
        
        return df
    
    def _encode_categorical(self, X: pd.DataFrame, fit: bool = True) -> np.ndarray:
        """
        Apply Label Encoding to categorical features.
        
        As per the paper: "Label Encoding maps each symbolic feature 
        category to a corresponding integer."
        """
        X = X.copy()
        
        for col in X.columns:
            if X[col].dtype == 'object' or X[col].dtype.name == 'category':
                if fit:
                    self.label_encoders[col] = LabelEncoder()
                    X[col] = self.label_encoders[col].fit_transform(X[col].astype(str))
                else:
                    if col in self.label_encoders:
                        X[col] = self.label_encoders[col].transform(X[col].astype(str))
        
        return X.values
    
    @property
    def num_classes(self) -> int:
        """Return number of classes"""
        return len(self.label_encoder.classes_)
    
    @property
    def class_names(self) -> List[str]:
        """Return class names"""
        return self.label_encoder.classes_.tolist()


class SiameseTripletDataset(Dataset):
    """
    Dataset for Siamese Network Triplet training.
    
    Generates triplets of samples:
    - anchor: A sample from a random class
    - positive: Another sample from the same class as anchor
    - negative: A sample from a different class
    
    Reference: Section 3.1 of the paper (Improved to Triplet)
    """
    
    def __init__(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        num_triplets: Optional[int] = None,
        balance_by_class: bool = True
    ):
        """
        Args:
            features: Feature array
            labels: Label array
            num_triplets: Number of triplets to generate (default: len(features))
        """
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels = labels
        self.num_triplets = num_triplets if num_triplets else len(features)
        self.balance_by_class = balance_by_class
        
        # Group indices by class
        self.class_indices = {}
        for idx, label in enumerate(labels):
            if label not in self.class_indices:
                self.class_indices[label] = []
            self.class_indices[label].append(idx)
        
        self.unique_labels = list(self.class_indices.keys())
        
    def __len__(self) -> int:
        return self.num_triplets
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns an anchor, positive, and negative sample.
        
        Returns:
            (anchor, positive, negative)
        """
        # Imbalance-aware anchor sampling:
        # If enabled, sample class first so minority classes are seen more often.
        if self.balance_by_class:
            label_anchor = random.choice(self.unique_labels)
            idx_anchor = random.choice(self.class_indices[label_anchor])
        else:
            idx_anchor = random.randint(0, len(self.features) - 1)
            label_anchor = self.labels[idx_anchor]
        
        # Select positive sample from same class (could be same as anchor, but ideally different if enough exist)
        idx_positive = random.choice(self.class_indices[label_anchor])
        
        # Select negative sample from different class
        other_labels = [l for l in self.unique_labels if l != label_anchor]
        if len(other_labels) > 0:
            label_negative = random.choice(other_labels)
            idx_negative = random.choice(self.class_indices[label_negative])
        else:
            # Fallback (edge case if only one class exists, completely invalid for triplet but safe-guard)
            idx_negative = random.choice(self.class_indices[label_anchor])
        
        return self.features[idx_anchor], self.features[idx_positive], self.features[idx_negative]


class IntrusionDataset(Dataset):
    """
    Dataset for intrusion detection classification.
    
    Used for training Predecessor and Successor models.
    """
    
    def __init__(self, features: np.ndarray, labels: np.ndarray):
        """
        Args:
            features: Feature array (already encoded by Siamese network)
            labels: Label array
        """
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)
        
    def __len__(self) -> int:
        return len(self.features)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.features[idx], self.labels[idx]
