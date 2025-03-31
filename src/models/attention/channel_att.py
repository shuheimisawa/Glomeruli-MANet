# src/models/attention/channel_att.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class ChannelAttention(nn.Module):
    """
    Channel Attention Module (SE-like)
    Focuses on important feature channels by weighting channels adaptively.
    """
    def __init__(self, in_channels, reduction_ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        # Shared MLP for both pooling outputs
        self.mlp = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction_ratio, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction_ratio, in_channels, kernel_size=1, bias=False)
        )
        
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        # Average pooling branch
        avg_pool = self.avg_pool(x)
        avg_out = self.mlp(avg_pool)
        
        # Max pooling branch
        max_pool = self.max_pool(x)
        max_out = self.mlp(max_pool)
        
        # Combine both branches
        out = avg_out + max_out
        scale = self.sigmoid(out)
        
        return x * scale
