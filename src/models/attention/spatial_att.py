# src/models/attention/spatial_att.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class SpatialAttention(nn.Module):
    """
    Spatial Attention Module
    Focuses on important spatial regions by weighting spatial positions adaptively.
    """
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        # Compute channel-wise statistics
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        
        # Concatenate along channel dimension
        out = torch.cat([avg_out, max_out], dim=1)
        
        # Apply convolution and sigmoid activation
        out = self.conv(out)
        scale = self.sigmoid(out)
        
        return x * scale
