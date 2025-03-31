#!/usr/bin/env python3
import os
import sys
import logging
import argparse
import yaml
from pathlib import Path
from typing import Dict, List, Any, Optional
import pandas as pd
from tqdm import tqdm
import shutil

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from src.data.patch_extractor import GlomeruliPatchExtractor
from src.utils.slide_utils import SlideReader
from src.utils.qupath_utils import QuPathAnnotationParser

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('preprocessing.log')
    ]
)
logger = logging.getLogger(__name__)

def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to config file
        
    Returns:
        Configuration dictionary
    """
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        logger.info(f"Loaded configuration from {config_path}")
        return config
    except Exception as e:
        logger.error(f"Failed to load configuration: {str(e)}")
        raise

def prepare_dataset_splits(
    output_dir: Path,
    slide_annotation_pairs: List[Dict[str, str]],
    train_ratio: float = 0.7,
    val_ratio: float = 0.1,
    test_ratio: float = 0.2,
    random_seed: int = 42
) -> Dict[str, List[Dict[str, str]]]:
    """
    Split the dataset into train, validation, and test sets.
    
    Args:
        output_dir: Output directory for splits
        slide_annotation_pairs: List of dictionaries containing slide and annotation paths
        train_ratio: Proportion for training set
        val_ratio: Proportion for validation set
        test_ratio: Proportion for test set
        random_seed: Random seed for reproducibility
        
    Returns:
        Dictionary with train, val, and test splits
    """
    import numpy as np
    
    # Check ratios
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-10, "Ratios must sum to 1"
    
    # Set random seed
    np.random.seed(random_seed)
    
    # Shuffle slide list
    indices = np.arange(len(slide_annotation_pairs))
    np.random.shuffle(indices)
    
    # Calculate split indices
    n_train = int(len(indices) * train_ratio)
    n_val = int(len(indices) * val_ratio)
    
    # Split indices
    train_indices = indices[:n_train]
    val_indices = indices[n_train:n_train + n_val]
    test_indices = indices[n_train + n_val:]
    
    # Create splits
    splits = {
        'train': [slide_annotation_pairs[i] for i in train_indices],
        'val': [slide_annotation_pairs[i] for i in val_indices],
        'test': [slide_annotation_pairs[i] for i in test_indices]
    }
    
    # Create directories for each split
    for split_name in splits.keys():
        split_dir = output_dir / split_name
        for class_name in QuPathAnnotationParser.CLASS_MAPPING.keys():
            (split_dir / class_name).mkdir(parents=True, exist_ok=True)
    
    # Save split information
    split_info = []
    for split_name, pairs in splits.items():
        for pair in pairs:
            split_info.append({
                'slide': Path(pair['slide']).name,
                'annotation': Path(pair['annotation']).name,
                'split': split_name
            })
    
    # Save as CSV
    pd.DataFrame(split_info).to_csv(output_dir / 'dataset_splits.csv', index=False)
    
    logger.info(f"Dataset split: train={len(splits['train'])}, "
                f"val={len(splits['val'])}, test={len(splits['test'])}")
                
    return splits

def copy_patches_to_splits(patch_dir: Path, output_dir: Path, dataset_splits: Dict[str, List[Dict[str, str]]]):
    """
    Copy extracted patches to train/val/test directories.
    
    Args:
        patch_dir: Directory containing extracted patches
        output_dir: Base output directory
        dataset_splits: Dictionary with dataset splits
    """
    # Create slide-to-split mapping
    slide_to_split = {}
    for split_name, pairs in dataset_splits.items():
        for pair in pairs:
            slide_name = Path(pair['slide']).stem
            slide_to_split[slide_name] = split_name
            
    logger.info(f"Slides in splits: {list(slide_to_split.keys())}")
    
    # Get class name mapping - handle case sensitivity issues
    class_dirs = {d.name: d for d in patch_dir.glob('*') if d.is_dir()}
    logger.info(f"Found class directories: {list(class_dirs.keys())}")
    
    # Create output directories with the exact same names as source
    for split_name in ['train', 'val', 'test']:
        for class_name in class_dirs.keys():
            (output_dir / split_name / class_name).mkdir(parents=True, exist_ok=True)
    
    # Copy files to appropriate split directories
    counts = {split_name: {} for split_name in ['train', 'val', 'test']}
    
    for class_name, class_dir in class_dirs.items():
        logger.info(f"Processing class: {class_name}")
        
        for patch_path in class_dir.glob('*.png'):
            # Extract slide name properly - assuming format like "tpath_001_glom01_class.png"
            parts = patch_path.stem.split('_')
            if len(parts) >= 2:
                # Reconstruct the full slide name (e.g., "tpath_001")
                slide_name = f"{parts[0]}_{parts[1]}"
                
                if slide_name in slide_to_split:
                    split_name = slide_to_split[slide_name]
                    dest_path = output_dir / split_name / class_name / patch_path.name
                    
                    # Copy file
                    shutil.copy2(patch_path, dest_path)
                    
                    # Count files
                    if class_name not in counts[split_name]:
                        counts[split_name][class_name] = 0
                    counts[split_name][class_name] += 1
                else:
                    logger.warning(f"Slide {slide_name} not found in any split")
            else:
                logger.warning(f"Unexpected filename format: {patch_path.name}")
    
    logger.info(f"Patch distribution by split: {counts}")

def main():
    parser = argparse.ArgumentParser(description='Preprocess whole slide images for glomeruli classification')
    parser.add_argument('--config', type=str, required=True, help='Path to configuration file')
    parser.add_argument('--n_slides', type=int, default=None, help='Number of slides to process (for testing)')
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Input/output paths
    data_root = Path(config['data']['root'])
    wsi_dir = data_root / config['data']['raw_dir']
    annotation_dir = data_root / config['data']['annotation_dir']
    output_dir = data_root / config['data']['processed_dir']
    
    # Create output directories
    patch_dir = output_dir / 'patches'
    patch_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all slide and annotation files
    slide_paths = list(wsi_dir.glob('*.svs'))
    logger.info(f"Found {len(slide_paths)} slide files in {wsi_dir}")
    
    # Check if we should limit the number of slides
    if args.n_slides is not None:
        slide_paths = slide_paths[:args.n_slides]
        logger.info(f"Limiting to {args.n_slides} slides for testing")
    
    # Match annotations with slides
    slide_annotation_pairs = []
    for slide_path in slide_paths:
        slide_name = slide_path.stem
        # Look for annotation file with "_annotations" suffix
        annotation_path = annotation_dir / f"{slide_name}_annotations.json"
        
        if annotation_path.exists():
            slide_annotation_pairs.append({
                'slide': str(slide_path),
                'annotation': str(annotation_path)
            })
            logger.info(f"Matched slide {slide_name} with annotation {annotation_path.name}")
        else:
            logger.warning(f"No annotation found for slide {slide_name}")
    
    logger.info(f"Matched {len(slide_annotation_pairs)} slides with annotations")
    
    # Extract patches from slides
    extractor = GlomeruliPatchExtractor(
        output_dir=str(patch_dir),
        patch_size=config['preprocessing']['patch_size'],
        magnification=config['preprocessing']['magnification'],
        context_factor=config['preprocessing']['context_factor'],
        n_workers=config['preprocessing']['num_workers']
    )
    
    # Process all slides
    results = {}
    for pair in tqdm(slide_annotation_pairs, desc="Processing slides"):
        try:
            counts = extractor.process_slide(
                slide_path=pair['slide'],
                annotation_path=pair['annotation'],
                save_thumbnails=True
            )
            results[Path(pair['slide']).stem] = counts
        except Exception as e:
            logger.error(f"Error processing {pair['slide']}: {str(e)}")
            results[Path(pair['slide']).stem] = {"error": str(e)}
    
    # Save results
    pd.DataFrame.from_dict(results, orient='index').to_csv(output_dir / 'patch_extraction_summary.csv')
    
    # Create train/val/test splits
    splits = prepare_dataset_splits(
        output_dir=output_dir,
        slide_annotation_pairs=slide_annotation_pairs,
        train_ratio=config['data']['train_ratio'],
        val_ratio=config['data']['val_ratio'],
        test_ratio=config['data']['test_ratio'],
        random_seed=config['training']['random_seed']
    )
    
    # Copy patches to split directories
    copy_patches_to_splits(patch_dir, output_dir, splits)
    
    logger.info("Preprocessing completed successfully")

if __name__ == "__main__":
    main()