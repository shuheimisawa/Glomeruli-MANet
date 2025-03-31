# src/models/segmentation/losses.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

class DiceLoss(nn.Module):
    """
    Dice loss for segmentation tasks.
    """
    def __init__(self, smooth: float = 1.0, reduction: str = 'mean'):
        """
        Initialize Dice loss.
        
        Args:
            smooth: Smoothing factor to avoid division by zero
            reduction: Reduction method ('mean', 'sum', 'none')
        """
        super().__init__()
        self.smooth = smooth
        self.reduction = reduction
        
    def forward(self, 
               pred: torch.Tensor, 
               target: torch.Tensor,
               eps: float = 1e-7) -> torch.Tensor:
        """
        Compute Dice loss.
        
        Args:
            pred: Predicted tensor (B, C, H, W)
            target: Target tensor (B, C, H, W) or (B, H, W)
            eps: Small epsilon to avoid division by zero
            
        Returns:
            Dice loss value
        """
        # Make sure target is one-hot if not already
        if target.dim() == 3:
            target = target.unsqueeze(1)
            
        if pred.dim() == 3:
            pred = pred.unsqueeze(1)
            
        # Flatten tensors
        pred_flat = pred.contiguous().view(-1)
        target_flat = target.contiguous().view(-1)
        
        # Compute intersection and union
        intersection = (pred_flat * target_flat).sum()
        union = pred_flat.sum() + target_flat.sum()
        
        # Compute Dice coefficient
        dice = (2. * intersection + self.smooth) / (union + self.smooth + eps)
        
        # Compute loss
        loss = 1 - dice
        
        if self.reduction == 'mean':
            return loss
        elif self.reduction == 'sum':
            return loss * target.size(0)
        else:  # 'none'
            return loss
            
class BCEDiceLoss(nn.Module):
    """
    Combined Binary Cross-Entropy and Dice Loss.
    """
    def __init__(self, 
                 bce_weight: float = 0.5, 
                 dice_weight: float = 0.5,
                 reduction: str = 'mean'):
        """
        Initialize combined loss.
        
        Args:
            bce_weight: Weight for BCE loss
            dice_weight: Weight for Dice loss
            reduction: Reduction method
        """
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.bce = nn.BCEWithLogitsLoss(reduction=reduction)
        self.dice = DiceLoss(reduction=reduction)
        
    def forward(self, 
               pred: torch.Tensor, 
               target: torch.Tensor) -> torch.Tensor:
        """
        Compute combined loss.
        
        Args:
            pred: Predicted tensor
            target: Target tensor
            
        Returns:
            Combined loss value
        """
        return self.bce_weight * self.bce(pred, target) + \
               self.dice_weight * self.dice(torch.sigmoid(pred), target)
               
class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance in segmentation.
    """
    def __init__(self, 
                 alpha: float = 0.25, 
                 gamma: float = 2.0,
                 reduction: str = 'mean'):
        """
        Initialize Focal loss.
        
        Args:
            alpha: Weighting factor for positive examples
            gamma: Focusing parameter
            reduction: Reduction method
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        
    def forward(self, 
               pred: torch.Tensor, 
               target: torch.Tensor) -> torch.Tensor:
        """
        Compute Focal loss.
        
        Args:
            pred: Predicted tensor
            target: Target tensor
            
        Returns:
            Focal loss value
        """
        # Apply sigmoid if not already
        if pred.max() > 1 or pred.min() < 0:
            pred_prob = torch.sigmoid(pred)
        else:
            pred_prob = pred
            
        # Flatten tensors
        pred_prob = pred_prob.view(-1)
        target = target.view(-1)
        
        # Compute focal loss
        # For positive examples: alpha * (1-p)^gamma * log(p)
        # For negative examples: (1-alpha) * p^gamma * log(1-p)
        bce = F.binary_cross_entropy(pred_prob, target, reduction='none')
        p_t = pred_prob * target + (1 - pred_prob) * (1 - target)
        focal_weight = self.alpha * target + (1 - self.alpha) * (1 - target)
        focal_weight = focal_weight * (1 - p_t).pow(self.gamma)
        
        loss = focal_weight * bce
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss
