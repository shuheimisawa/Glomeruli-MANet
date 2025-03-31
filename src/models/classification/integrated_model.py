# src/models/classification/integrated_model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from ..attention.maunit import MAUnit

class MANet(nn.Module):
    """
    Multiple Attention Network (MANet) for glomeruli classification
    Combines multiple attention mechanisms to enhance feature extraction.
    """
    def __init__(self, 
                 in_channels=3, 
                 num_classes=4, 
                 base_filters=64,
                 dropout_rate=0.5):
        super(MANet, self).__init__()
        
        # Initial convolution
        self.initial_conv = nn.Sequential(
            nn.Conv2d(in_channels, base_filters, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(base_filters),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        
        # Stage 1
        self.stage1 = nn.Sequential(
            MAUnit(base_filters, base_filters, reduction_ratio=16),
            MAUnit(base_filters, base_filters, reduction_ratio=16),
            MAUnit(base_filters, base_filters, reduction_ratio=16)
        )
        
        # Stage 2
        self.stage2 = nn.Sequential(
            MAUnit(base_filters, base_filters*2, reduction_ratio=16),
            MAUnit(base_filters*2, base_filters*2, reduction_ratio=16),
            MAUnit(base_filters*2, base_filters*2, reduction_ratio=16)
        )
        
        # Stage 3
        self.stage3 = nn.Sequential(
            MAUnit(base_filters*2, base_filters*4, reduction_ratio=16),
            MAUnit(base_filters*4, base_filters*4, reduction_ratio=16),
            MAUnit(base_filters*4, base_filters*4, reduction_ratio=16)
        )
        
        # Global average pooling and classification head
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(base_filters*4, num_classes)
        )
        
        # Initialize weights
        self._initialize_weights()
        
    def forward(self, x):
        # Feature extraction
        x = self.initial_conv(x)
        
        # Multi-stage feature refinement with attention
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        
        # Global pooling and classification
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        
        return x
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)