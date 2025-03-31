# src/models/attention/cbam.py
import torch
import torch.nn as nn
from .channel_att import ChannelAttention
from .spatial_att import SpatialAttention

class CBAM(nn.Module):
    """
    Convolutional Block Attention Module (CBAM)
    Combines channel and spatial attention mechanisms in sequence.
    """
    def __init__(self, in_channels, reduction_ratio=16, spatial_kernel_size=7):
        super(CBAM, self).__init__()
        self.channel_att = ChannelAttention(in_channels, reduction_ratio)
        self.spatial_att = SpatialAttention(spatial_kernel_size)
        
    def forward(self, x):
        # Apply channel attention first
        x = self.channel_att(x)
        
        # Then apply spatial attention
        x = self.spatial_att(x)
        
        return x
