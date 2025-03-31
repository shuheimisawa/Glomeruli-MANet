# src/models/segmentation/unet.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Optional, Tuple, Union

class ConvBlock(nn.Module):
    """
    Double convolution block for UNet++ with batch normalization.
    """
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        return x

class NestedUNet(nn.Module):
    """
    UNet++ implementation for glomeruli segmentation.
    
    Deep supervision allows the network to output segmentation maps at multiple depths.
    """
    
    def __init__(self, 
                in_channels: int = 3, 
                out_channels: int = 1, 
                filters: List[int] = [32, 64, 128, 256, 512],
                deep_supervision: bool = True):
        """
        Initialize UNet++ model.
        
        Args:
            in_channels: Number of input channels (3 for RGB)
            out_channels: Number of output channels (1 for binary segmentation)
            filters: Number of filters at each level
            deep_supervision: Whether to use deep supervision
        """
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.filters = filters
        self.deep_supervision = deep_supervision
        
        # Create encoder pathway (downsampling)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoders = nn.ModuleList([
            ConvBlock(in_channels, filters[0]),
            ConvBlock(filters[0], filters[1]),
            ConvBlock(filters[1], filters[2]),
            ConvBlock(filters[2], filters[3]),
            ConvBlock(filters[3], filters[4])
        ])
        
        # Create nested decoder paths
        # Format: conv{i}{j} where i is the level, j is the nested block index
        self.decoders = nn.ModuleDict()
        
        # Upsampling operations for each nested block
        self.upsamples = nn.ModuleDict()
        
        # Create the nested decoder blocks and upsamples
        for i in range(1, 5):  # 4 levels (excluding the bottom)
            for j in range(1, i+1):  # j nested blocks per level
                # Input channels calculation for nested blocks
                if j == 1:
                    # First block in each level gets input from encoder and upsampled lower level
                    in_ch = filters[i-1] + filters[i]
                else:
                    # Other blocks get input from previous block in the same level and
                    # all corresponding blocks from previous levels
                    in_ch = filters[i-1] * j + filters[i]
                
                # Create upsampling operation
                self.upsamples[f'up{i}{j}'] = nn.ConvTranspose2d(
                    filters[i], filters[i-1], kernel_size=2, stride=2
                )
                
                # Create decoder block
                self.decoders[f'conv{i}{j}'] = ConvBlock(in_ch, filters[i-1])
        
        # Deep supervision outputs
        if deep_supervision:
            self.segmentation_heads = nn.ModuleList([
                nn.Conv2d(filters[0], out_channels, kernel_size=1)
                for _ in range(4)  # One for each depth level
            ])
        else:
            self.final_conv = nn.Conv2d(filters[0], out_channels, kernel_size=1)
    
    def forward(self, x: torch.Tensor) -> Union[torch.Tensor, List[torch.Tensor]]:
        """
        Forward pass of UNet++.
        
        Args:
            x: Input tensor
            
        Returns:
            Segmentation output tensor or list of tensors for deep supervision
        """
        # Encoder path outputs
        encoder_features = []
        
        # Store encoder outputs
        e1 = self.encoders[0](x)
        encoder_features.append(e1)
        
        e2 = self.encoders[1](self.pool(e1))
        encoder_features.append(e2)
        
        e3 = self.encoders[2](self.pool(e2))
        encoder_features.append(e3)
        
        e4 = self.encoders[3](self.pool(e3))
        encoder_features.append(e4)
        
        e5 = self.encoders[4](self.pool(e4))
        
        # Nested decoder path
        # Store all decoder outputs for skip connections
        decoder_outputs = {}
        
        # Level 1 (first decoder level)
        decoder_outputs['x01'] = encoder_features[0]  # Encoder output at level 0
        up1 = self.upsamples['up11'](e2)
        decoder_outputs['x11'] = self.decoders['conv11'](
            torch.cat([up1, encoder_features[0]], dim=1)
        )
        
        # Level 2
        up2 = self.upsamples['up21'](e3)
        decoder_outputs['x21'] = self.decoders['conv21'](
            torch.cat([up2, encoder_features[1]], dim=1)
        )
        
        up = self.upsamples['up22'](decoder_outputs['x21'])
        decoder_outputs['x12'] = self.decoders['conv12'](
            torch.cat([up, decoder_outputs['x01'], decoder_outputs['x11']], dim=1)
        )
        
        # Level 3
        up3 = self.upsamples['up31'](e4)
        decoder_outputs['x31'] = self.decoders['conv31'](
            torch.cat([up3, encoder_features[2]], dim=1)
        )
        
        up = self.upsamples['up32'](decoder_outputs['x31'])
        decoder_outputs['x22'] = self.decoders['conv22'](
            torch.cat([up, decoder_outputs['x21']], dim=1)
        )
        
        up = self.upsamples['up33'](decoder_outputs['x22'])
        decoder_outputs['x13'] = self.decoders['conv13'](
            torch.cat([up, decoder_outputs['x01'], decoder_outputs['x11'], decoder_outputs['x12']], dim=1)
        )
        
        # Level 4
        up4 = self.upsamples['up41'](e5)
        decoder_outputs['x41'] = self.decoders['conv41'](
            torch.cat([up4, encoder_features[3]], dim=1)
        )
        
        up = self.upsamples['up42'](decoder_outputs['x41'])
        decoder_outputs['x32'] = self.decoders['conv32'](
            torch.cat([up, decoder_outputs['x31']], dim=1)
        )
        
        up = self.upsamples['up43'](decoder_outputs['x32'])
        decoder_outputs['x23'] = self.decoders['conv23'](
            torch.cat([up, decoder_outputs['x21'], decoder_outputs['x22']], dim=1)
        )
        
        up = self.upsamples['up44'](decoder_outputs['x23'])
        decoder_outputs['x14'] = self.decoders['conv14'](
            torch.cat([
                up, decoder_outputs['x01'], decoder_outputs['x11'],
                decoder_outputs['x12'], decoder_outputs['x13']
            ], dim=1)
        )
        
        # Output(s)
        if self.deep_supervision:
            # Return outputs at different depths
            outputs = []
            outputs.append(self.segmentation_heads[0](decoder_outputs['x11']))
            outputs.append(self.segmentation_heads[1](decoder_outputs['x12']))
            outputs.append(self.segmentation_heads[2](decoder_outputs['x13']))
            outputs.append(self.segmentation_heads[3](decoder_outputs['x14']))
            return outputs
        else:
            # Return only the final output
            return self.final_conv(decoder_outputs['x14'])
    
    def use_checkpointing(self):
        """
        Use gradient checkpointing to reduce memory usage.
        Useful for training on large images.
        """
        self.checkpointing = True
