# LoopNet

LoopNet: A Multitasking Few-Shot Learning Approach for Loop Closure in Large Scale SLAM

## Installation

1. Clone the repository:
```bash
git clone https://github.com/RovisLab/LoopNet
cd LoopNet
```

2. Create and activate a virtual environment (optional but recommended):
```bash
python -m venv venv
# On Windows
venv\Scripts\activate
# On Linux/Mac
source venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

### Training
To train the model:
```bash
python LoopNet.py
```

Optional arguments:
- `--config`: Path to config file (default: configs/config.yaml)
- `--resume`: Path to checkpoint to resume training

### Evaluation
To evaluate a trained model:
```bash
python evaluate.py --checkpoint checkpoints/best_model.pth
```

## Model Architecture

The model combines:
- DISK feature extractor for robust feature detection
- ResNet18 backbone for deep feature extraction
- Custom fusion layers for feature combination
- Contrastive learning approach for similarity learning

### Key Components:
1. DISK Feature Extractor
2. ResNet18 Backbone
3. Feature Fusion Module
4. Classification Head
5. Similarity Head


## Output Analysis

The evaluation produces:
1. Confidence distribution plots
2. Feature distribution analysis
3. Class distribution analysis
4. Detailed predictions for each test image
5. Visualization samples


## Project Structure
```
├── checkpoints/          # Model checkpoints directory
├── configs/             
│   └── config.yaml      # Configuration file
├── data/
│   └── dataset.py       # Dataset handling
├── models/
│   ├── disk.py          # DISK feature extractor
│   ├── loss.py          # Loss functions
│   ├── resnet_disk.py   # Main model architecture
│   └── __init__.py
├── outputs/             # Evaluation outputs
├── utils/
│   ├── config.py        # Configuration utilities
│   ├── training.py      # Training functions
│   ├── visualization.py # Visualization utilities
│   └── __init__.py
├── LoopNet.py            # Training script
├── evaluate.py         # Evaluation script
└── requirements.txt    # Project dependencies
```

## Requirements
```
torch>=2.0.0
torchvision>=0.15.0
kornia>=0.7.0
numpy>=1.21.0
Pillow>=9.0.0
matplotlib>=3.5.0
seaborn>=0.11.0
scikit-learn>=1.0.0
tqdm>=4.65.0
pyyaml>=6.0.0
```

## License

GPL-3.0 license

## Citation





