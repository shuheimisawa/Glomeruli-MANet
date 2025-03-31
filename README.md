# Glomeruli-MANet: Deep Learning for Glomeruli Sclerosis Classification

## Project Overview
This project implements a deep learning framework for classifying glomeruli states in renal biopsy images. The system focuses on identifying four distinct states:

1. **Normal**: Healthy glomeruli without sclerosis
2. **Partially Sclerotic**: Glomeruli showing early signs of sclerosis
3. **Sclerotic**: Fully sclerotic glomeruli
4. **Uncertain**: Cases where classification is ambiguous

The system is designed to process whole slide images (WSI) from renal biopsies, automating the classification of glomeruli to support pathologists in diagnosing kidney conditions.

## System Requirements
- Python 3.11
- PyTorch
- AMD Radeon RX 6700 XT GPU support
- AMD Ryzen 9 5900X 12-core processor
- 32 GB RAM
- 400 GB available SSD storage

## Project Structure
```
glomeruli-manet/
├── configs/                      # Configuration files
│   ├── default.yaml              # Base settings
│   ├── classification.yaml       # Classification model settings
│   └── training.yaml             # Training hyperparameters
├── data/                         # Data directories
│   ├── raw/                      # Original WSI (.svs) files
│   ├── annotations/              # QuPath GeoJSON annotations
│   └── processed/                # Processed images and masks
│       ├── train/
│       │   ├── normal/           # Normal glomeruli images
│       │   ├── partially_sclerotic/  # Partially sclerotic images
│       │   ├── sclerotic/        # Fully sclerotic images
│       │   └── uncertain/        # Uncertain cases
│       ├── val/
│       └── test/
├── logs/                         # Training logs
├── experiments/                  # Model checkpoints and results
│   ├── classification/           # Classification model checkpoints
│   └── results/                  # Evaluation results and visualizations
├── notebooks/                    # Jupyter notebooks for analysis
├── scripts/                      # Execution scripts
├── src/                          # Source code
│   ├── data/                     # Data processing modules
│   ├── models/                   # Model implementations
│   ├── training/                 # Training logic
│   └── utils/                    # Utility functions
├── tests/                        # Unit and integration tests
├── app/                          # Application for deployment
├── docker/                       # Containerization files
├── environment.yml               # Conda environment
├── requirements.txt              # Python dependencies
└── README.md                     # Project documentation
```

## Setup and Installation

1. Clone the repository:
```bash
git clone https://github.com/shuheimisawa/Glomeruli-MANet.git
cd Glomeruli-MANet
```

2. Create and activate the conda environment:
```bash
conda env create -f environment.yml
conda activate glomeruli-manet
```

3. Install the package:
```bash
pip install -e .
```

## Data Preprocessing
The system processes whole slide images (WSI) from renal biopsies:

1. **WSI Loading**: Load slides using OpenSlide library
2. **Annotation Parsing**: Parse QuPath GeoJSON annotations
3. **Patch Extraction**: Extract glomeruli patches at appropriate magnification
4. **Normalization**: Apply Z-score normalization for consistent input

Example:
```bash
python scripts/preprocess_wsi.py --config configs/default.yaml
```

## Model Architecture

### Glomeruli Classifier
The model uses a CNN-based architecture with attention mechanisms:

- **Backbone**: Pre-trained ResNet50 with custom classification head
- **Attention**: CBAM (Convolutional Block Attention Module) for focusing on relevant regions
- **Classification**: 4-class softmax output for sclerosis states

Key features:
- Multi-scale feature extraction
- Attention-based feature refinement
- Class-balanced training
- Uncertainty estimation

## Training Pipeline
The training process follows these steps:

1. **Data Preparation**:
   - Extract glomeruli patches from WSIs
   - Apply data augmentation
   - Balance classes if necessary

2. **Model Training**:
   - Train on balanced dataset
   - Validate on separate set
   - Monitor class-wise performance

Training parameters are configurable through YAML files, with defaults optimized for medical image classification.

## Evaluation and Metrics
The system uses multiple metrics to evaluate performance:
- Overall accuracy
- Per-class precision, recall, and F1-score
- Confusion matrix
- ROC curves for each class
- Uncertainty analysis

## Deployment
The application can be deployed using:
- Containerization: Docker for consistent deployment
- Web Interface: Simple UI for pathologists
- API: RESTful API for integration with other systems

## Next Steps
1. Data collection and annotation
2. Model training and validation
3. Performance evaluation
4. User interface development
5. Deployment and testing

## Contact
For questions or collaboration opportunities, please contact misawa.shuhei@hotmail.com.