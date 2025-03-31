# Glomeruli-MANet: Production-Grade Deep Learning for Renal Biopsy Analysis

## Project Overview
This project implements a state-of-the-art deep learning framework for renal biopsy wet slide image analysis. The system follows a two-stage approach:

1. **Pre-segmentation Module**: UNet++ architecture for glomeruli segmentation
2. **Classification Module**: Multiple Attention Network (MANet) for classifying:
   - Deposition region (capillary wall, mesangial)
   - Deposition appearance (granular, lumps, linear)

The system is designed to process wet slide images from SVS slides annotated with QuPath, automating the identification and classification of glomeruli to support pathologists in diagnosing glomerulonephritis.

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
│   ├── segmentation.yaml         # Segmentation model settings
│   ├── classification.yaml       # Classification model settings
│   └── training.yaml             # Training hyperparameters
├── data/                         # Data directories
│   ├── raw/                      # Original WSI (.svs) files
│   ├── annotations/              # QuPath GeoJSON annotations
│   └── processed/                # Processed images and masks
│       ├── train/
│       │   ├── images/           # Training images
│       │   ├── masks/            # Segmentation masks
│       │   └── metadata.json     # Training patch metadata
│       ├── val/
│       └── test/
├── logs/                         # Training logs
├── experiments/                  # Model checkpoints and results
│   ├── segmentation/             # Segmentation model checkpoints
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
The system uses QuPath's GeoJSON annotations to extract and process regions of interest from SVS slides:

1. **WSI Loading**: Load slides using OpenSlide library
2. **Annotation Parsing**: Parse QuPath GeoJSON annotations
3. **Patch Extraction**: Extract patches at the appropriate magnification
4. **Normalization**: Apply Z-score normalization for consistent input

Example:
```bash
python scripts/preprocess_wsi.py --config configs/default.yaml
```

## Model Architecture

### UNet++ Segmentation Module
The segmentation model uses UNet++ with a pre-trained ResNet34 encoder backbone for effective glomeruli isolation:
- Deep supervision for improved feature learning
- Skip connections to preserve spatial information
- Dice + BCE combined loss function

### MANet Classification Module
The classification module employs multiple attention mechanisms:
- Dynamic Selection Mechanism: Adapts to varying glomerular sizes using different convolution kernels
- Channel Attention: Focuses on important features through channel relationships
- Spatial Attention: Improves localization within the image

The model includes parallel branches for region and appearance classification, with the results fused to generate comprehensive reports.

## Training Pipeline
The training process follows these steps:

1. **Pre-segmentation Training**:
   - Input: Raw immunofluorescence images
   - Output: Segmentation masks for glomeruli

2. **Classification Training**:
   - Input: Segmented glomeruli
   - Output: Region and appearance classifications

Training parameters are fully configurable through YAML files, with sensible defaults based on the published research.

## Evaluation and Metrics
The system uses multiple metrics to evaluate performance:
- Segmentation: Dice coefficient, IoU
- Classification: Accuracy, precision, recall, F1-score, AUC
- Visualization: Grad-CAM for model interpretability

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