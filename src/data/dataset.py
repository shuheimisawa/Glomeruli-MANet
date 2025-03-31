# src/data/dataset.py
import torch
from torch.utils.data import Dataset
import numpy as np
import cv2
from pathlib import Path
import albumentations as A
from albumentations.pytorch import ToTensorV2

class GlomeruliDataset(Dataset):
    """
    Dataset class for glomeruli patch classification.
    """
    def __init__(self, 
                 root_dir,
                 split='train', 
                 transform=None,
                 class_map=None):
        """
        Initialize the dataset.
        
        Args:
            root_dir: Root directory containing processed patches
            split: Data split ('train', 'val', 'test')
            transform: Optional transform to apply
            class_map: Optional mapping from class names to indices
        """
        self.root_dir = Path(root_dir)
        self.split = split
        
        # Default class mapping if not provided
        if class_map is None:
            self.class_map = {
                'Normal': 0,
                'Partially Sclerotic': 1,
                'Sclerotic': 2,
                'Uncertain': 3
            }
        else:
            self.class_map = class_map
            
        # Find all image paths and their corresponding labels
        self.samples = []
        split_dir = self.root_dir / split
        
        for class_dir in split_dir.glob('*'):
            if not class_dir.is_dir():
                continue
                
            class_name = class_dir.name
            # Get class index, handling case sensitivity
            for key in self.class_map.keys():
                if key.lower() == class_name.lower():
                    class_idx = self.class_map[key]
                    break
            else:
                continue  # Skip if class not found
                
            # Add all images from this class
            for img_path in class_dir.glob('*.png'):
                self.samples.append((str(img_path), class_idx))
        
        # Default transformations if none provided
        if transform is None:
            if split == 'train':
                self.transform = A.Compose([
                    A.RandomRotate90(),
                    A.Flip(),
                    A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5),
                    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                    ToTensorV2(),
                ])
            else:
                self.transform = A.Compose([
                    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                    ToTensorV2(),
                ])
        else:
            self.transform = transform
            
    def __len__(self):
        return len(self.samples)
        
    def __getitem__(self, idx):
        img_path, class_idx = self.samples[idx]
        
        # Read image
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Apply transformations
        if self.transform:
            transformed = self.transform(image=img)
            img = transformed['image']
            
        return img, class_idx
