"""
BERT-of-Theseus Knowledge Distillation Module

Implements the improved BERT-of-Theseus method for knowledge distillation
as described in Section 3.2-3.3 of the paper.

Key concepts:
- Module replacement mechanism (Equations 5-6)
- MSE loss function (Equation 7)
- Gradient optimization for faster convergence (Equations 8-13)
- Bernoulli distribution for module selection

The method compresses the Predecessor (teacher) into the Successor (student)
through module-by-module replacement training.

Reference: Sections 3.2-3.3 of Wang et al. (2024)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional, List
import math

def knowledge_distillation_loss(student_logits, teacher_logits, labels, criterion_mse, T=3.0, alpha=0.5):
    """
    Computes a combination of MSE loss and KL Divergence loss (Logit Distillation).
    """
    mse_loss = criterion_mse(student_logits, labels)
    
    soft_student = F.log_softmax(student_logits / T, dim=1)
    with torch.no_grad():
        soft_teacher = F.softmax(teacher_logits / T, dim=1)
        
    # kl_loss = nn.KLDivLoss(reduction='batchmean')(soft_student, soft_teacher) * (T * T)
    kl_loss = nn.KLDivLoss(reduction='batchmean')(soft_student, soft_teacher)
    
    return alpha * mse_loss + (1.0 - alpha) * kl_loss


class MixModule(nn.Module):
    """
    A Mix module that combines Predecessor and Successor modules.
    
    As per Section 3.2 (Equations 5-6):
        r_{i+1} ~ Bernoulli(p)                                    (Eq. 5)
        y_{i+1} = r_{i+1} * prd_i(y_i) + (1 - r_{i+1}) * scc_i(y_i)  (Eq. 6)
    
    During training, randomly selects between Predecessor and Successor modules.
    
    Args:
        predecessor_module: Module from Predecessor model
        successor_module: Module from Successor model
        replacement_rate: Probability of using Successor (p in Eq. 5)
    """
    
    def __init__(
        self,
        predecessor_module: nn.Module,
        successor_module: nn.Module,
        replacement_rate: float = 0.5
    ):
        super(MixModule, self).__init__()
        self.predecessor_module = predecessor_module
        self.successor_module = successor_module
        self.replacement_rate = replacement_rate
        
        # Store outputs for gradient optimization
        self.predecessor_output = None
        self.successor_output = None
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with probabilistic module selection.
        
        Equation 5: r_{i+1} ~ Bernoulli(p)
        Equation 6: y_{i+1} = r_{i+1} * prd_i(y_i) + (1 - r_{i+1}) * scc_i(y_i)
        
        Args:
            x: Input tensor
            
        Returns:
            Output tensor (from either Predecessor or Successor)
        """
        # Compute outputs from both modules
        with torch.no_grad():
            self.predecessor_output = self.predecessor_module(x)
        
        self.successor_output = self.successor_module(x)
        
        if self.training:
            # Equation 5: Sample from Bernoulli distribution
            # p = replacement_rate (probability of using Successor)
            r = torch.bernoulli(torch.tensor(1.0 - self.replacement_rate)).item()
            
            if r == 1.0:
                # Use Predecessor output (detached to freeze gradients)
                return self.predecessor_output.detach()
            else:
                # Use Successor output
                return self.successor_output
        else:
            # During inference, use Successor
            return self.successor_output

class SoftMixModule(nn.Module):
    """
    Soft Mix module that interpolates between Predecessor and Successor.
    
    Instead of hard binary replacement (0 or 1), this uses a continuous
    blending weight (alpha) that gradually shifts focus from Teacher to Student.
    
    Formula: y_{i+1} = (1 - alpha) * prd_i(y_i) + alpha * scc_i(y_i)
    """
    
    def __init__(
        self,
        predecessor_module: nn.Module,
        successor_module: nn.Module,
        alpha: float = 0.0
    ):
        super(SoftMixModule, self).__init__()
        self.predecessor_module = predecessor_module
        self.successor_module = successor_module
        self.alpha = alpha  # 0.0 = Use only Predecessor, 1.0 = Use only Successor
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Get Predecessor output (Teacher)
        with torch.no_grad():
            predecessor_output = self.predecessor_module(x)
            
        # Get Successor output (Student)
        successor_output = self.successor_module(x)
        
        if self.training:
            # SOFT REPLACEMENT
            # Detach predecessor_output so gradients only flow to Successor
            mixed_output = (1.0 - self.alpha) * predecessor_output.detach() + self.alpha * successor_output
            return mixed_output
        else:
            # During inference, use Successor
            return successor_output

class OptimizedMixModule(nn.Module):
    """
    Optimized Mix module with gradient-aware replacement rate selection.
    
    Implements the distillation optimization from Section 3.3 (Equations 8-13).
    
    The optimization analyzes gradient propagation and adaptively selects
    the replacement rate to maximize gradient magnitude while avoiding
    vanishing gradients.
    
    Key insight from paper:
    - When r_{i+1} = 1: gradient is minimum (vanishing)
    - When r_{i+1} = 0: gradient = 2(scc_i(y_i) - y_label)
    - Optimal r_{i+1} depends on comparing |scc_i(y_i) - y_label| and |Γ|
    
    Args:
        predecessor_module: Module from Predecessor model
        successor_module: Module from Successor model
        base_replacement_rate: Base probability of using Successor
    """
    
    def __init__(
        self,
        predecessor_module: nn.Module,
        successor_module: nn.Module,
        base_replacement_rate: float = 0.5
    ):
        super(OptimizedMixModule, self).__init__()
        
        self.predecessor_module = predecessor_module
        self.successor_module = successor_module
        self.base_replacement_rate = base_replacement_rate
        
        # Cache for gradient optimization
        self.predecessor_output = None
        self.successor_output = None
        self.use_optimization = True
    
    def compute_optimal_r(
        self, 
        scc_output: torch.Tensor, 
        prd_output: torch.Tensor,
        y_label: torch.Tensor
    ) -> float:
        """
        Compute optimal replacement rate based on gradient analysis.
        
        Implements Equations 11-12 from the paper.
        
        IMPORTANT FIX: The paper's equations assume outputs and labels are in the same space.
        However, intermediate module outputs have shape [B, Seq, Embed] while labels are [B].
        We must reduce scc and prd to scalar representations per sample before comparison.
        
        Equation 11: ∂L/∂θ^s_i = 2[(scc_i(y_i) - prd_i(y_i))r²_{i+1} + 
                     (prd_i(y_i) - 2scc_i(y_i) + y_label)r_{i+1} + 
                     scc_i(y_i) - y_label] * ∂scc_i(y_i)/∂θ^s_i
        
        Equation 12: r_{i+1} = 
            0                                               if |scc_i(y_i) - y_label| >= |Γ|
            -(prd_i(y_i) - 2scc_i(y_i) + y_label) / 
             (2(scc_i(y_i) - prd_i(y_i)))                   if |scc_i(y_i) - y_label| < |Γ|
        
        Args:
            scc_output: Successor module output [B, Seq, Embed]
            prd_output: Predecessor module output [B, Seq, Embed]
            y_label: Target label [B]
            
        Returns:
            Optimal replacement rate
        """
        # FIX: Reduce 3D tensors to 1D (per-sample scalar) using Global Average Pooling
        # scc_output shape: [Batch, Seq_len, Embed_dim] -> [Batch]
        # prd_output shape: [Batch, Seq_len, Embed_dim] -> [Batch]
        if scc_output.dim() == 3:
            scc_mean = scc_output.mean(dim=(1, 2))  # [Batch]
            prd_mean = prd_output.mean(dim=(1, 2))  # [Batch]
        elif scc_output.dim() == 2:
            scc_mean = scc_output.mean(dim=1)  # [Batch]
            prd_mean = prd_output.mean(dim=1)  # [Batch]
        else:
            scc_mean = scc_output.flatten()
            prd_mean = prd_output.flatten()
        
        # y_label is [Batch] containing class indices
        y = y_label.float()
        
        # Now all tensors are 1D [Batch], can perform element-wise operations
        # Compute |scc_i(y_i) - y_label|
        scc_minus_label = (scc_mean - y).abs().mean().item()
        
        # Compute coefficients for quadratic equation
        # a = (scc - prd)
        # b = (prd - 2*scc + y_label)
        a = (scc_mean - prd_mean).mean().item()
        b = (prd_mean - 2 * scc_mean + y).mean().item()
        
        # Avoid division by zero
        if abs(a) < 1e-8:
            return 0.0
        
        # Compute extreme point of quadratic
        r_extreme = -b / (2 * a)
        
        # Compute Γ (extreme value of the gradient term)
        # When r = r_extreme, the gradient term reaches its extreme
        gamma = a * r_extreme ** 2 + b * r_extreme + scc_minus_label
        
        # Equation 12: Select optimal r
        if scc_minus_label >= abs(gamma):
            # Case A: r = 0 gives larger gradient
            return 0.0
        else:
            # Case C: Use r_extreme, but ensure it's in valid range
            return max(0.0, min(1.0, r_extreme))
    
    def forward(
        self, 
        x: torch.Tensor, 
        y_label: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass with optimized module selection.
        
        Uses gradient-aware selection when y_label is provided.
        
        Args:
            x: Input tensor
            y_label: Optional target labels for gradient optimization
            
        Returns:
            Output tensor
        """
        # Compute outputs from both modules
        with torch.no_grad():
            self.predecessor_output = self.predecessor_module(x)
        
        self.successor_output = self.successor_module(x)
        
        if self.training:
            if self.use_optimization and y_label is not None:
                # Use gradient-optimized replacement rate
                optimal_r = self.compute_optimal_r(
                    self.successor_output.detach(),
                    self.predecessor_output.detach(),
                    y_label
                )
                
                # Sample based on optimal r (inverted because r=0 means use Successor)
                use_successor = np.random.random() > optimal_r
            else:
                # Standard Bernoulli sampling
                use_successor = np.random.random() < (1 - self.base_replacement_rate)
            
            if use_successor:
                return self.successor_output
            else:
                return self.predecessor_output.detach()
        else:
            return self.successor_output

class ProjectedMixModule(nn.Module):
    """
    Mix module equipped with a Heterogeneous Feature Alignment (Projection) layer.
    
    Instead of directly returning the Successor's output (which may be in a different
    mathematical space due to PoolFormer vs ViT architecture), this module projects
    the Successor's output into the Predecessor's space before mixing.
    """
    
    def __init__(
        self,
        predecessor_module: nn.Module,
        successor_module: nn.Module,
        embed_dim: int,
        replacement_rate: float = 0.5
    ):
        super(ProjectedMixModule, self).__init__()
        self.predecessor_module = predecessor_module
        self.successor_module = successor_module
        self.replacement_rate = replacement_rate
        
        # Non-linear projection to map PoolFormer features into ViT space
        self.projection = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim)
        )
        
        self.predecessor_output = None
        self.successor_output = None
        self.projected_output = None
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            self.predecessor_output = self.predecessor_module(x)
            
        self.successor_output = self.successor_module(x)
        self.projected_output = self.projection(self.successor_output)
        
        if self.training:
            # Equation 5: Sample from Bernoulli distribution
            r = torch.bernoulli(torch.tensor(1.0 - self.replacement_rate)).item()
            
            if r == 1.0:
                # Use Predecessor output (detached to freeze gradients)
                return self.predecessor_output.detach()
            else:
                # Use Project Successor output
                return self.projected_output
        else:
            # During inference, use Projected Successor output
            return self.projected_output


class MixModel(nn.Module):
    """
    Mix Model combining Predecessor and Successor for BERT-of-Theseus training.
    
    As per Section 3.2 (Fig. 4):
    - The Mix model comprises multiple submodules (Mix_B_i)
    - Each submodule contains corresponding modules from Predecessor and Successor
    - During training, randomly selects between modules using Bernoulli sampling
    
    Args:
        predecessor: Full Predecessor model
        successor: Full Successor model
        replacement_rate: Initial probability of using Successor
        use_optimization: Whether to use gradient optimization
    """
    
    def __init__(
        self,
        predecessor: nn.Module,
        successor: nn.Module,
        replacement_rate: float = 0.5,
        use_optimization: bool = True,
        use_soft_replacement: bool = True,
        use_projection: bool = True
    ):
        super(MixModel, self).__init__()
        
        self.predecessor = predecessor
        self.successor = successor
        self.num_modules = len(predecessor.modules_list)
        self.use_optimization = use_optimization
        self.use_soft_replacement = use_soft_replacement
        self.use_projection = use_projection
        
        ''' Freeze Predecessor parameters
        for param in predecessor.parameters():
            param.requires_grad = False
        '''
        successor.patch_embed.weight.data.copy_(predecessor.patch_embed.projection.weight.data)
        successor.patch_embed.bias.data.copy_(predecessor.patch_embed.projection.bias.data)
        # Create Mix modules
        if self.use_projection:
            embed_dim = predecessor.embed_dim
            self.mix_modules = nn.ModuleList([
                ProjectedMixModule(
                    predecessor.get_module(i),
                    successor.get_module(i),
                    embed_dim,
                    replacement_rate
                )
                for i in range(self.num_modules)
            ])
        elif self.use_soft_replacement:
            self.mix_modules = nn.ModuleList([
                SoftMixModule(
                    predecessor.get_module(i),
                    successor.get_module(i),
                    alpha=0.0 # Bắt đầu từ 0 (100% Teacher)
                )
                for i in range(self.num_modules)
            ])
        elif use_optimization:
            self.mix_modules = nn.ModuleList([
                OptimizedMixModule(
                    predecessor.get_module(i),
                    successor.get_module(i),
                    replacement_rate
                )
                for i in range(self.num_modules)
            ])
        else:
            self.mix_modules = nn.ModuleList([
                MixModule(
                    predecessor.get_module(i),
                    successor.get_module(i),
                    replacement_rate
                )
                for i in range(self.num_modules)
            ])
        
        # Use Successor's patch embedding and classification head
        self.patch_embed = successor.patch_embed
        self.pos_embed = successor.pos_embed
        self.norm = successor.norm
        self.pool = successor.pool
        self.classifier = successor.classifier
    
    def forward(
        self, 
        x: torch.Tensor, 
        y_label: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass through Mix model.
        
        Args:
            x: Input tensor (batch_size, channels, height, width)
            y_label: Optional target labels for gradient optimization
            
        Returns:
            Classification logits
        """
        # Patch Embedding
        x = self.patch_embed(x)
        batch_size = x.shape[0]
        x = x.flatten(2).transpose(1, 2)
        
        # Add Positional Embedding
        x = x + self.pos_embed
        
        # Pass through Mix modules
        for mix_module in self.mix_modules:
            if self.use_optimization and y_label is not None:
                x = mix_module(x, y_label)
            else:
                x = mix_module(x)
        
        # Final processing
        x = self.norm(x)
        x = x.transpose(1, 2)
        x = self.pool(x).squeeze(-1)
        x = self.classifier(x)
        
        return x
    def update_alpha(self, new_alpha: float):
        """Update alpha for SoftMix modules"""
        for mix_module in self.mix_modules:
            if hasattr(mix_module, 'alpha'):
                mix_module.alpha = new_alpha
    
    def update_replacement_rate(self, new_rate: float):
        """Update replacement rate for all Mix modules"""
        for mix_module in self.mix_modules:
            if hasattr(mix_module, 'replacement_rate'):
                mix_module.replacement_rate = new_rate
            elif hasattr(mix_module, 'base_replacement_rate'):
                mix_module.base_replacement_rate = new_rate


class BERTOfTheseus:
    """
    BERT-of-Theseus Knowledge Distillation Framework.
    
    Implements the full knowledge distillation pipeline:
    1. Pre-train Predecessor model
    2. Pre-train Successor model
    3. Module replacement training using Mix model
    4. Fine-tune Successor model
    
    Uses MSE loss (Equation 7):
        L = mean[(y - label)²]
    
    Args:
        predecessor: Predecessor model (teacher)
        successor: Successor model (student)
        device: Device for training
        initial_replacement_rate: Starting replacement rate (default: 0.5)
        use_optimization: Whether to use gradient optimization
    """
    
    def __init__(
        self,
        predecessor: nn.Module,
        successor: nn.Module,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        initial_replacement_rate: float = 0.5,
        use_optimization: bool = True,
        use_kd_loss: bool = True,
        kd_T: float = 3.0,
        kd_alpha: float = 0.5,
        use_projection: bool = True
    ):
        self.predecessor = predecessor.to(device)
        self.successor = successor.to(device)
        self.device = device
        self.initial_replacement_rate = initial_replacement_rate
        self.use_optimization = use_optimization
        self.use_kd_loss = use_kd_loss
        self.kd_T = kd_T
        self.kd_alpha = kd_alpha
        
        # Get number of classes from classifier
        self.num_classes = successor.classifier.out_features
        
        # Create Mix model
        self.mix_model = MixModel(
            predecessor,
            successor,
            initial_replacement_rate,
            use_optimization,
            use_soft_replacement=False,
            use_projection=use_projection
        ).to(device)
        
        # Loss function: MSE (Equation 7) as per paper Section 3.2
        # Paper states: "In the replacement training process, this paper utilizes 
        # the Mean Squared Error (MSE) loss function, as shown in Eq. (7):
        # L = mean[(y - label)²]"
        self.criterion = nn.MSELoss()
    
    def _to_onehot(self, target: torch.Tensor) -> torch.Tensor:
        """
        Convert class indices to one-hot encoding for MSE loss.
        
        Required because MSE loss needs targets with same shape as outputs.
        
        Args:
            target: Class indices tensor [Batch]
            
        Returns:
            One-hot encoded tensor [Batch, num_classes]
        """
        return F.one_hot(target, num_classes=self.num_classes).float()
        
    def pre_train_predecessor(
        self,
        train_loader,
        epochs: int = 50,
        learning_rate: float = 0.0001
    ) -> List[float]:
        """
        Pre-train the Predecessor model.
        
        Args:
            train_loader: DataLoader for training data
            epochs: Number of training epochs
            learning_rate: Learning rate
            
        Returns:
            List of training losses
        """
        print("Pre-training Predecessor model...")
        
        self.predecessor.train()
        optimizer = torch.optim.Adam(self.predecessor.parameters(), lr=learning_rate)
        
        losses = []
        for epoch in range(epochs):
            epoch_loss = 0.0
            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(self.device), target.to(self.device)
                
                optimizer.zero_grad()
                output = self.predecessor(data)
                # Convert target to one-hot for MSE loss (Equation 7)
                target_onehot = self._to_onehot(target)
                loss = self.criterion(output, target_onehot)
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
            
            avg_loss = epoch_loss / len(train_loader)
            losses.append(avg_loss)
            
            if (epoch + 1) % 10 == 0:
                print(f"  Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
        
        return losses
    
    def pre_train_successor(
        self,
        train_loader,
        epochs: int = 50,
        learning_rate: float = 0.0001
    ) -> List[float]:
        """
        Pre-train the Successor model.
        
        Args:
            train_loader: DataLoader for training data
            epochs: Number of training epochs
            learning_rate: Learning rate
            
        Returns:
            List of training losses
        """
        print("Pre-training Successor model...")
        
        self.successor.train()
        optimizer = torch.optim.Adam(self.successor.parameters(), lr=learning_rate)
        
        losses = []
        for epoch in range(epochs):
            epoch_loss = 0.0
            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(self.device), target.to(self.device)
                
                optimizer.zero_grad()
                output = self.successor(data)
                # Convert target to one-hot for MSE loss (Equation 7)
                target_onehot = self._to_onehot(target)
                loss = self.criterion(output, target_onehot)
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
            
            avg_loss = epoch_loss / len(train_loader)
            losses.append(avg_loss)
            
            if (epoch + 1) % 10 == 0:
                print(f"  Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
        
        return losses
    
    def module_replacement_training(
        self,
        train_loader,
        epochs: int = 250,
        learning_rate: float = 0.0001,
        schedule_replacement: bool = True
    ) -> List[float]:
        """
        Perform module replacement training (BERT-of-Theseus).
        
        As per Section 3.2-3.3:
        - Freeze Predecessor parameters
        - Train Mix model with module replacement
        - Gradually increase replacement rate
        
        Args:
            train_loader: DataLoader for training data
            epochs: Number of training epochs (default: 250 as per paper)
            learning_rate: Learning rate
            schedule_replacement: Whether to increase replacement rate over time
            
        Returns:
            List of training losses
        """
        print("Module replacement training (BERT-of-Theseus)...")
        
        # Freeze Predecessor
        for param in self.predecessor.parameters():
            param.requires_grad = False
        
        # Only train Successor parameters
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.mix_model.parameters()),
            lr=learning_rate
        )
        
        losses = []
        for epoch in range(epochs):
            # # Schedule replacement rate (linear increase)
            # if schedule_replacement:
            #     current_rate = min(0.9, self.initial_replacement_rate + 
            #                      (0.9 - self.initial_replacement_rate) * epoch / epochs)
            #     self.mix_model.update_replacement_rate(current_rate)
            
            # self.mix_model.train()
            # Schedule replacement rate OR alpha (linear increase)
            if schedule_replacement:
                if getattr(self, 'use_soft_replacement', False):
                    # Soft replacement: alpha trượt từ 0.0 (100% Teacher) đến 1.0 (100% Student)
                    current_alpha = min(1.0, epoch / epochs)
                    self.mix_model.update_alpha(current_alpha)
                    display_metric = f"| Alpha: {current_alpha:.2f}"
                else:
                    # Hard replacement (Code gốc)
                    current_rate = min(0.9, self.initial_replacement_rate + 
                                     (0.9 - self.initial_replacement_rate) * epoch / epochs)
                    self.mix_model.update_replacement_rate(current_rate)
                    display_metric = f"| Rep Rate: {current_rate:.2f}"
            else:
                display_metric = ""
                
            self.mix_model.train()
            epoch_loss = 0.0
            
            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(self.device), target.to(self.device)
                
                optimizer.zero_grad()
                
                if self.use_optimization:
                    output = self.mix_model(data, target)
                else:
                    output = self.mix_model(data)
                
                # Convert target to one-hot for MSE loss (Equation 7)
                target_onehot = self._to_onehot(target)
                if self.use_kd_loss:
                    with torch.no_grad():
                        teacher_logits = self.predecessor(data)
                    loss = knowledge_distillation_loss(
                        output, teacher_logits, target_onehot, 
                        self.criterion, self.kd_T, self.kd_alpha
                    )
                else:
                    loss = self.criterion(output, target_onehot)
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
            
            avg_loss = epoch_loss / len(train_loader)
            losses.append(avg_loss)
            
            # if (epoch + 1) % 50 == 0:
            #     print(f"  Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}, "
            #           f"Replacement Rate: {current_rate:.2f}" if schedule_replacement 
            #           else f"  Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
            if (epoch + 1) % 50 == 0:
                print(f"  Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f} {display_metric}")
        
        return losses
    
    def fine_tune_successor(
        self,
        train_loader,
        val_loader,
        max_epochs: int = 100,
        learning_rate: float = 0.0001,
        patience: int = 10
    ) -> Tuple[List[float], List[float]]:
        """
        Fine-tune the Successor model until validation metrics stop improving.
        
        As per Section 3.5: "During the fine-tuning phase, the Successor is 
        trained until the validation dataset indicators no longer rise."
        
        Args:
            train_loader: DataLoader for training data
            val_loader: DataLoader for validation data
            max_epochs: Maximum number of epochs
            learning_rate: Learning rate
            patience: Early stopping patience
            
        Returns:
            Tuple of (training_losses, validation_losses)
        """
        print("Fine-tuning Successor model...")
        
        self.successor.train()
        optimizer = torch.optim.Adam(self.successor.parameters(), lr=learning_rate)
        
        train_losses = []
        val_losses = []
        best_val_loss = float('inf')
        patience_counter = 0
        best_state = None
        
        for epoch in range(max_epochs):
            # Training
            self.successor.train()
            epoch_loss = 0.0
            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(self.device), target.to(self.device)
                
                optimizer.zero_grad()
                output = self.successor(data)
                # Convert target to one-hot for MSE loss (Equation 7)
                target_onehot = self._to_onehot(target)
                if self.use_kd_loss:
                    with torch.no_grad():
                        teacher_logits = self.predecessor(data)
                    loss = knowledge_distillation_loss(
                        output, teacher_logits, target_onehot, 
                        self.criterion, self.kd_T, self.kd_alpha
                    )
                else:
                    loss = self.criterion(output, target_onehot)
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
            
            train_loss = epoch_loss / len(train_loader)
            train_losses.append(train_loss)
            
            # Validation
            self.successor.eval()
            val_loss = 0.0
            with torch.no_grad():
                for data, target in val_loader:
                    data, target = data.to(self.device), target.to(self.device)
                    output = self.successor(data)
                    # Convert target to one-hot for MSE loss (Equation 7)
                    target_onehot = self._to_onehot(target)
                    loss = self.criterion(output, target_onehot)
                    val_loss += loss.item()
            
            val_loss /= len(val_loader)
            val_losses.append(val_loss)
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_state = self.successor.state_dict().copy()
            else:
                patience_counter += 1
            
            if (epoch + 1) % 10 == 0:
                print(f"  Epoch {epoch+1}/{max_epochs}, "
                      f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            
            if patience_counter >= patience:
                print(f"  Early stopping at epoch {epoch+1}")
                break
        
        # Restore best model
        if best_state is not None:
            self.successor.load_state_dict(best_state)
        
        return train_losses, val_losses
    
    def full_training_pipeline(
        self,
        train_loader,
        val_loader,
        pre_train_epochs: int = 50,
        replacement_epochs: int = 250,
        fine_tune_epochs: int = 100,
        learning_rate: float = 0.0001
    ) -> dict:
        """
        Execute the full BT-TPF training pipeline.
        
        Steps (from Section 3.5):
        1. Pre-train Predecessor model
        2. Pre-train Successor model
        3. Module replacement training
        4. Fine-tune Successor model
        
        Args:
            train_loader: DataLoader for training data
            val_loader: DataLoader for validation data
            pre_train_epochs: Epochs for pre-training (default: 50)
            replacement_epochs: Epochs for module replacement (default: 250)
            fine_tune_epochs: Max epochs for fine-tuning (default: 100)
            learning_rate: Learning rate (default: 0.0001)
            
        Returns:
            Dictionary containing all training history
        """
        history = {}
        
        # Step 1: Pre-train Predecessor
        history['predecessor_loss'] = self.pre_train_predecessor(
            train_loader, pre_train_epochs, learning_rate
        )
        
        # Step 2: Pre-train Successor
        history['successor_pretrain_loss'] = self.pre_train_successor(
            train_loader, pre_train_epochs, learning_rate
        )
        
        # Step 3: Module replacement training
        history['replacement_loss'] = self.module_replacement_training(
            train_loader, replacement_epochs, learning_rate
        )
        
        # Step 4: Fine-tune Successor
        train_losses, val_losses = self.fine_tune_successor(
            train_loader, val_loader, fine_tune_epochs, learning_rate
        )
        history['fine_tune_train_loss'] = train_losses
        history['fine_tune_val_loss'] = val_losses
        
        return history
    
    def get_successor(self) -> nn.Module:
        """Return the trained Successor model"""
        return self.successor
    
    def get_predecessor(self) -> nn.Module:
        """Return the Predecessor model"""
        return self.predecessor
