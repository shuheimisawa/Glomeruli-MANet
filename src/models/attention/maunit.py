# src/models/attention/maunit.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from .channel_att import ChannelAttention
from .spatial_att import SpatialAttention

class MAUnit(nn.Module):
    """
    Multiple Attention Unit (MAUnit)
    Combines dynamic selection, channel attention, and spatial attention mechanisms.
    """
    def __init__(self, in_channels, out_channels, reduction_ratio=16, L=32):
        super(MAUnit, self).__init__()
        
        # Dynamic selection mechanism (multi-scale convolutions)
        self.conv3x3 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.conv5x5 = nn.Conv2d(in_channels, out_channels, kernel_size=5, stride=1, padding=2)
        
        # Channel attention for dynamic selection
        mid_channels = max(out_channels // reduction_ratio, L)
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(out_channels, mid_channels, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels * 2, kernel_size=1, bias=False)
        )
        
        # Spatial attention
        self.spatial_att = SpatialAttention(kernel_size=7)
        
        # Batch normalization and activation
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
        # Skip connection (with 1x1 conv if needed for channel matching)
        self.skip = nn.Sequential()
        if in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        # Skip connection
        residual = self.skip(x)
        
        # Dynamic selection mechanism (multi-scale feature extraction)
        conv3 = self.conv3x3(x)
        conv5 = self.conv5x5(x)
        
        # Fusion of features
        fused = conv3 + conv5
        fused = self.bn(fused)
        
        # Channel attention weights
        att_weights = self.fc(fused)
        att_weights = att_weights.reshape(-1, 2, fused.size(1), 1, 1)
        att_weights = F.softmax(att_weights, dim=1)
        
        # Apply attention weights
        scaled_conv3 = conv3 * att_weights[:, 0]
        scaled_conv5 = conv5 * att_weights[:, 1]
        out = scaled_conv3 + scaled_conv5
        
        # Apply spatial attention
        out = self.spatial_att(out)
        
        # Add skip connection and apply ReLU
        out = out + residual
        out = self.relu(out)
        
        return out