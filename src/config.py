"""
Configuration Module for BT-TPF Framework

Contains all hyperparameters as specified in Section 5.2 of the paper.

Reference: Section 5.2 of Wang et al. (2024)
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class SiameseConfig:
    """
    Configuration for Siamese Network.
    
    From Section 5.2:
    - Hidden layer neurons: 5
    - Margin in Triplet Loss: 0.3
    - Output dimension: 36 (for 6×6 reshape)
    """
    hidden_dim: int = 5
    output_dim: int = 36  # For 6×6 reshape
    margin: float = 0.3


@dataclass
class PatchEmbeddingConfig:
    """
    Configuration for Patch Embedding.
    
    From Section 5.2:
    - Convolution step (stride): 2
    - Convolution kernel size: 2
    - Number of convolution kernels (embed_dim): 8
    """
    stride: int = 2
    kernel_size: int = 2
    embed_dim: int = 8


@dataclass
class PredecessorConfig:
    """
    Configuration for Predecessor (ViT-based) model.
    
    From Section 5.2:
    - Number of heads in Multi-Head Attention: 2
    - MLP middle layer neurons: 4× token sequence length (num_patches)
      For 6×6 input with patch_size=2: num_patches = 9, so MLP hidden = 4×9 = 36
    - 9 layers (3 modules × 3 blocks)
    """
    input_channels: int = 1
    input_size: int = 6  # 6×6 feature map
    embed_dim: int = 8
    num_modules: int = 3
    blocks_per_module: int = 3  # Total 9 layers
    num_heads: int = 2
    mlp_ratio: int = 4  # MLP hidden = mlp_ratio × num_patches = 4 × 9 = 36
    dropout: float = 0.1


@dataclass
class SuccessorConfig:
    """
    Configuration for Successor (PoolFormer-based) model.
    
    From Section 5.2:
    - MLP middle layer neurons: 1 (minimal for lightweight)
    - 3 layers (3 modules × 1 block)
    """
    input_channels: int = 1
    input_size: int = 6  # 6×6 feature map
    embed_dim: int = 8
    num_modules: int = 3
    blocks_per_module: int = 1  # Total 3 layers
    mlp_hidden_dim: int = 1  # Exactly 1 neuron as per paper Section 5.2
    dropout: float = 0.1


@dataclass
class TrainingConfig:
    """
    Training configuration.
    
    From Section 5.2:
    - Optimizer: Adam
    - Activation: Tanh
    - Batch size: 1024
    - Initial learning rate: 0.0001
    - Pre-training epochs: 50
    - Hybrid training epochs: 250
    - Fine-tuning: until validation indicators no longer rise
    """
    optimizer: str = 'adam'
    activation: str = 'tanh'
    batch_size: int = 1024
    learning_rate: float = 0.0001
    pretrain_epochs: int = 50
    replacement_epochs: int = 250
    finetune_epochs: int = 100
    early_stopping_patience: int = 10
    
    # Knowledge distillation
    initial_replacement_rate: float = 0.5
    use_gradient_optimization: bool = True
    
    # Imbalance handling
    use_weighted_sampler: bool = True
    weighted_sampler_power: float = 1.0


@dataclass
class DataConfig:
    """
    Data configuration.
    
    From Section 4:
    - Train/test split: 75%/25%
    - CIC-IDS2017: 78 features → 36 (encoded)
    - TON_IoT: 43 features → 36 (encoded)
    """
    test_size: float = 0.25
    random_state: int = 42
    encoded_dim: int = 36  # After Siamese encoding
    
    # Siamese pair sampling strategy
    siamese_balance_by_class: bool = True


@dataclass
class BTPTFConfig:
    """
    Complete configuration for BT-TPF framework.
    """
    siamese: SiameseConfig = None
    patch_embedding: PatchEmbeddingConfig = None
    predecessor: PredecessorConfig = None
    successor: SuccessorConfig = None
    training: TrainingConfig = None
    data: DataConfig = None
    
    # Device configuration
    device: str = 'cuda'  # Will fallback to CPU if CUDA unavailable
    
    def __post_init__(self):
        if self.siamese is None:
            self.siamese = SiameseConfig()
        if self.patch_embedding is None:
            self.patch_embedding = PatchEmbeddingConfig()
        if self.predecessor is None:
            self.predecessor = PredecessorConfig()
        if self.successor is None:
            self.successor = SuccessorConfig()
        if self.training is None:
            self.training = TrainingConfig()
        if self.data is None:
            self.data = DataConfig()


def get_cicids2017_config() -> BTPTFConfig:
    """
    Get configuration for CIC-IDS2017 dataset.
    
    CIC-IDS2017 has:
    - 78 features (before encoding)
    - 5 classes: Benign, GoldenEye, Hulk, Slowhttptest, Slowloris
    """
    config = BTPTFConfig()
    config.predecessor.num_classes = 5
    config.successor.num_classes = 5
    return config


def get_toniot_config() -> BTPTFConfig:
    """
    Get configuration for TON_IoT dataset.
    
    TON_IoT has:
    - 43 features (before encoding)
    - 10 classes: Normal, Backdoor, DDoS, DoS, Injection, 
                  MITM, Password, Ransomware, Scanning, XSS
    """
    config = BTPTFConfig()
    config.predecessor.num_classes = 10
    config.successor.num_classes = 10
    return config
