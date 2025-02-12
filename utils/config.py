import yaml
from pathlib import Path

def load_config(config_path: str = "configs/config.yaml"):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def get_transforms(config):
    """Create transform configurations from config"""
    from torchvision import transforms
    
    train_transform = transforms.Compose([
        transforms.Resize(config['transforms']['train']['resize']),
        transforms.RandomCrop(config['transforms']['train']['crop_size']),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(30),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=config['transforms']['train']['mean'],
            std=config['transforms']['train']['std']
        )
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize(config['transforms']['val']['resize']),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=config['transforms']['val']['mean'],
            std=config['transforms']['val']['std']
        )
    ])

    return train_transform, val_transform
