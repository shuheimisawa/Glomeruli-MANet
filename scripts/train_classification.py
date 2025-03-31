# scripts/train_classification.py
#!/usr/bin/env python3
import os
import sys
import logging
import argparse
import yaml
from pathlib import Path
import torch

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from src.data.dataset import GlomeruliDataset
from src.models.integrated_model import MANet
from src.training.classification_trainer import ClassificationTrainer
from src.utils.logger import setup_logger

def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def main():
    parser = argparse.ArgumentParser(description='Train glomeruli classification model')
    parser.add_argument('--config', type=str, required=True, help='Path to configuration file')
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Setup logging
    log_dir = Path(config['logging']['save_dir']) / 'logs'
    log_dir.mkdir(parents=True, exist_ok=True)
    setup_logger(log_file=str(log_dir / 'training.log'))
    logger = logging.getLogger(__name__)
    
    # Set random seed
    if 'random_seed' in config['training']:
        seed = config['training']['random_seed']
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
        logger.info(f"Set random seed to {seed}")
    
    # Create datasets
    logger.info("Creating datasets...")
    data_root = Path(config['data']['root']) / config['data']['processed_dir']
    
    train_dataset = GlomeruliDataset(
        root_dir=data_root,
        split='train'
    )
    
    val_dataset = GlomeruliDataset(
        root_dir=data_root,
        split='val'
    )
    
    logger.info(f"Train dataset size: {len(train_dataset)}")
    logger.info(f"Validation dataset size: {len(val_dataset)}")
    
    # Create model
    logger.info("Creating model...")
    model = MANet(
        in_channels=3,
        num_classes=len(train_dataset.class_map),
        base_filters=config['model']['base_filters'],
        dropout_rate=config['model']['dropout_rate']
    )
    
    logger.info(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Create trainer
    trainer = ClassificationTrainer(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        config=config
    )
    
    # Train model
    logger.info("Starting training...")
    trainer.train(epochs=config['training']['epochs'])
    
    logger.info("Training completed")

if __name__ == "__main__":
    main()