# Glomeruli-MANet: Multiple Attention CNN Framework for Glomeruli Segmentation and Classification

## Overview

Glomeruli-MANet is a deep learning framework designed for accurate segmentation and classification of glomeruli in renal biopsy immunofluorescence images. It implements a two-stage convolutional neural network approach with multiple attention mechanisms to achieve state-of-the-art performance in identifying both the deposition region and appearance of glomeruli.

The framework leverages a pre-segmentation module to first isolate glomerular objects, followed by a dedicated Multiple Attention Network (MANet) that processes these regions through parallel classification branches for deposition region and appearance classification.

## Key Features

- **Two-stage pipeline architecture**: Segmentation followed by specialized classification
- **Multiple attention mechanisms**: Combines channel and spatial attention for improved feature detection
- **Parallel classification branches**: Separate dedicated networks for deposition region and appearance classification
- **End-to-end training pipeline**: From raw WSI files with QuPath annotations to final classification
- **Comprehensive evaluation metrics**: Analysis of both segmentation and classification performance

## Project Structure

```
glomeruli-manet/
├── data/
│   ├── raw/                  # Directory for raw WSI (.svs) files
│   ├── annotations/          # QuPath annotation files (.geojson)
│   └── processed/            # Processed image patches and masks
│       ├── train/
│       │   ├── images/       # Training image patches
│       │   ├── masks/        # Training segmentation masks
│       │   └── metadata.json # Training patch annotations and labels
│       ├── val/              # Validation data (similar structure)
│       └── test/             # Test data (similar structure)
├── src/
│   ├── data/
│   │   ├── preprocessing.py  # WSI and annotation processing utilities
│   │   ├── patch_extractor.py # Extract patches from WSIs based on annotations
│   │   ├── dataset.py        # PyTorch dataset classes
│   │   └── augmentation.py   # Data augmentation strategies
│   ├── models/
│   │   ├── segmentation/
│   │   │   ├── unet.py       # U-Net segmentation network
│   │   │   └── losses.py     # Segmentation loss functions (Dice, BCE, etc.)
│   │   ├── attention/
│   │   │   ├── channel_att.py # Channel attention module
│   │   │   ├── spatial_att.py # Spatial attention module
│   │   │   └── cbam.py       # Combined attention module
│   │   ├── classification/
│   │   │   ├── region_branch.py # Deposition region classification network
│   │   │   └── appearance_branch.py # Appearance classification network
│   │   └── integrated_model.py # Complete two-stage model integration
│   ├── training/
│   │   ├── segmentation_trainer.py # Trainer for segmentation module
│   │   ├── classification_trainer.py # Trainer for classification module
│   │   └── metrics.py        # Performance metrics calculation
│   └── utils/
│       ├── visualization.py  # Result visualization utilities
│       ├── slide_utils.py    # WSI reading and processing utilities
│       └── qupath_utils.py   # QuPath annotation parsing utilities
├── notebooks/
│   ├── data_exploration.ipynb # Explore dataset characteristics
│   ├── model_training.ipynb  # Step-by-step training process
│   └── results_analysis.ipynb # Analyze and visualize results
├── scripts/
│   ├── preprocess_wsi.py     # Convert WSIs and annotations to patches
│   ├── train_segmentation.py # Train segmentation module
│   ├── train_classification.py # Train classification module
│   ├── evaluate.py           # Comprehensive model evaluation
│   └── predict.py            # Run inference on new samples
├── experiments/
│   ├── segmentation/         # Segmentation model checkpoints and logs
│   ├── classification/       # Classification model checkpoints and logs
│   └── results/              # Evaluation results and visualizations
├── configs/
│   ├── segmentation.yaml     # Configuration for segmentation training
│   ├── classification.yaml   # Configuration for classification training
│   └── default.yaml          # Default configuration parameters
├── requirements.txt          # Python dependencies
└── environment.yml           # Conda environment specification
```

## Installation and Setup

### Requirements

- Python 3.8+
- CUDA-capable GPU (recommended at least 8GB VRAM)
- OpenSlide library for working with WSI files
- DirectML capable

# Create and activate a conda environment
conda env create -f environment.yml
conda activate glomeruli-manet

# Or use pip
pip install -r requirements.txt
```

## Data Preparation

### Input Data Format

The framework expects:
- Whole Slide Images (.svs files) in `data/raw/`
- QuPath annotations (.geojson files) in `data/annotations/`

### Preprocessing Pipeline

1. **Extract Annotated Regions**: Extract glomeruli regions from WSIs based on QuPath annotations
2. **Create Segmentation Masks**: Generate binary masks for glomeruli from annotations
3. **Split Data**: Divide into training, validation, and test sets with appropriate stratification
4. **Generate Metadata**: Create JSON files with deposition region and appearance labels

Run the preprocessing script:

```bash
python scripts/preprocess_wsi.py --wsi-dir data/raw/ --annotation-dir data/annotations/ --output-dir data/processed/ --patch-size 256 --overlap 64
```

## Model Architecture

### 1. Pre-Segmentation Module

The first stage is a U-Net based segmentation network:

- **Input**: Immunofluorescence image patches (256×256 pixels)
- **Output**: Binary segmentation mask for glomerular structures
- **Architecture**: U-Net with 4 encoding and 4 decoding blocks
- **Loss Function**: Combination of Binary Cross-Entropy and Dice loss

### 2. Multiple Attention Classification Module

The second stage consists of:

- **Base Feature Extractor**: CNN backbone (ResNet-50) for feature extraction
- **Attention Mechanisms**:
  - Channel attention: Highlights important feature maps
  - Spatial attention: Focuses on relevant spatial regions
  - Combined attention (CBAM): Integrates both attention types
- **Parallel Classification Branches**:
  - Region branch: Classifies deposition region (mesangial, capillary loop, etc.)
  - Appearance branch: Classifies deposition appearance (granular, linear, etc.)
- **Fusion Module**: Combines and weights outputs from both branches

## Training Process

### Phase 1: Segmentation Module Training

```bash
python scripts/train_segmentation.py --config configs/segmentation.yaml
```

Key parameters:
- Learning rate: 1e-4 with reduction on plateau
- Batch size: 16
- Epochs: 100
- Augmentations: Rotation, flipping, brightness/contrast variation

### Phase 2: Classification Module Training

```bash
python scripts/train_classification.py --config configs/classification.yaml --segmentation-model path/to/best/segmentation/model.pth
```

Key parameters:
- Learning rate: 1e-5 with cosine annealing
- Batch size: 32
- Epochs: 150
- Augmentations: Color jittering, Gaussian noise, elastic transform

## Evaluation

Run comprehensive evaluation on test data:

```bash
python scripts/evaluate.py --model-path path/to/final/model.pth --test-dir data/processed/test/
```

The evaluation includes:
- Segmentation metrics: Dice, Jaccard, Precision, Recall
- Classification metrics: Accuracy, F1-score, Precision, Recall (per class)
- Confusion matrices for both region and appearance classification
- Visualization of attention maps and segmentation masks

## Inference on New Data

For predicting on new WSI files:

```bash
python scripts/predict.py --model-path path/to/model.pth --wsi-path path/to/new/slide.svs --output-dir results/
```

## Results Visualization

The framework includes tools for visualizing:
- Segmentation masks
- Attention heatmaps
- Classification results overlaid on original images
- Confusion matrices

Example visualization code:

```python
from src.utils.visualization import visualize_results

# Load test image and run inference
visualize_results(image_path, model, output_path='visualization.png')
```

---

# Implementation Guide

This section provides detailed guidance for implementing the key components of the framework.

## Pre-Segmentation Module Implementation

The pre-segmentation module uses a U-Net architecture with the following specifications:

1. **Encoder**: Four downsampling blocks with double convolution (32, 64, 128, 256 filters)
2. **Bottleneck**: Double convolution with 512 filters
3. **Decoder**: Four upsampling blocks with skip connections (256, 128, 64, 32 filters)
4. **Output**: 1×1 convolution with sigmoid activation for binary segmentation

Implementation details:
- Use batch normalization after each convolution
- ReLU activation between layers
- Dropout (0.2) in bottleneck to prevent overfitting
- Kaiming weight initialization for all convolutional layers

## Multiple Attention Network Implementation

The MANet consists of:

1. **Base Network**: Modified ResNet-50 with first layer adapted for immunofluorescence input
2. **Attention Modules**:
   - Channel attention with reduction ratio 16
   - Spatial attention with kernel size 7
   - Combined attention (CBAM) after each residual block
3. **Feature Fusion**: Concatenation of global average pooling and max pooling features

## Classification Branches Implementation

Each classification branch follows:

1. **Feature Input**: Feature maps from the MANet (2048 features)
2. **Feature Refinement**: Two fully connected layers (1024, 512 neurons)
3. **Dropout**: 0.5 dropout rate between FC layers
4. **Output Layer**: Softmax activation with class-specific outputs

## Data Processing Implementation

The data processing pipeline includes:

1. **WSI Handling**: Use OpenSlide to read and extract patches
2. **Annotation Parsing**: Parse QuPath .geojson annotations to extract:
   - Glomeruli boundaries for segmentation masks
   - Deposition region labels (mesangial, capillary, etc.)
   - Appearance labels (granular, linear, etc.)
3. **Patch Extraction**: Extract patches at 40x magnification with adjustable overlap
4. **Normalization**: Standardize immunofluorescence intensity with target mean/std
5. **Augmentation**: Implement online data augmentation with Albumentations library

## Training Implementation Details

1. **Loss Functions**:
   - Segmentation: Weighted combination of BCE and Dice loss
   - Classification: Cross-entropy with class weights for imbalanced data
2. **Optimizers**:
   - Adam optimizer with weight decay (1e-5)
   - Learning rate scheduling with patience-based reduction
3. **Training Monitoring**:
   - TensorBoard integration for loss and metrics tracking
   - Model checkpointing based on validation performance
   - Early stopping with configurable patience

## Model Integration and End-to-End Pipeline

The complete pipeline integrates all components:

1. **Input Processing**: Normalize and preprocess input image
2. **Segmentation**: Generate binary mask of glomeruli regions
3. **Region Extraction**: Crop and resize segmented regions
4. **Feature Extraction**: Process through MANet with attention
5. **Parallel Classification**: Process through both classification branches
6. **Output Fusion**: Combine and format classification results
7. **Result Visualization**: Generate attention maps and overlay results

This framework handles the complete workflow from raw WSI files with QuPath annotations to final glomeruli classification with interpretable results.