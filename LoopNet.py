# main.py
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import logging
import argparse
from pathlib import Path

from data.dataset import CustomDataset
from models.resnet_disk import ResNet18WithDISK
from models.loss import ContrastiveLoss
from utils.config import load_config, get_transforms
from utils.training import Trainer
from utils.visualization import plot_training_progress

def setup_logging(config):
    """Setup logging configuration"""
    logging.basicConfig(
        level=config['output']['log_level'],
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)

def create_directories(config):
    """Create necessary directories"""
    for dir_path in [
        config['output']['checkpoints_dir'],
        config['output']['plots_dir'],
        config['output']['analysis_dir'],
        config['output']['sample_predictions_dir']
    ]:
        os.makedirs(dir_path, exist_ok=True)

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train the model')
    parser.add_argument('--config', default='configs/config.yaml', help='Path to config file')
    parser.add_argument('--resume', help='Path to checkpoint to resume from')
    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)
    
    # Setup logging and create directories
    logger = setup_logging(config)
    create_directories(config)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    try:
        # Setup data transforms
        train_transform, val_transform = get_transforms(config)
        
        # Create datasets
        train_dataset = CustomDataset(
            config['data']['train_path'],
            transform=train_transform,
            is_test=False
        )
        
        val_dataset = CustomDataset(
            config['data']['val_path'],
            transform=val_transform,
            is_test=False
        )
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=config['data']['batch_size'],
            shuffle=True,
            num_workers=config['data']['num_workers'],
            pin_memory=config['data']['pin_memory']
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=config['data']['batch_size'],
            shuffle=False,
            num_workers=config['data']['num_workers'],
            pin_memory=config['data']['pin_memory']
        )
        
        # Initialize model
        model = ResNet18WithDISK(
            num_classes=config['model']['num_classes'],
            device=device
        )
        
        # Setup loss functions
        criterion_cls = nn.CrossEntropyLoss()
        criterion_sim = ContrastiveLoss()
        
        # Setup optimizer
        optimizer = optim.AdamW(
            model.parameters(),
            lr=config['training']['learning_rate'],
            weight_decay=config['training']['weight_decay']
        )
        
        # Setup scheduler
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.2,
            patience=2,
            verbose=True
        )

        # Initialize trainer
        trainer = Trainer(args.config)
        
        # Resume from checkpoint if specified
        start_epoch = 0
        if args.resume:
            if os.path.isfile(args.resume):
                logger.info(f"Loading checkpoint '{args.resume}'")
                checkpoint = torch.load(args.resume, map_location=device)
                start_epoch = checkpoint['epoch']
                model.load_state_dict(checkpoint['model_state_dict'])
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                logger.info(f"Loaded checkpoint '{args.resume}' (epoch {checkpoint['epoch']})")
            else:
                logger.error(f"No checkpoint found at '{args.resume}'")
                return

        # Train model
        logger.info("Starting training...")
        train_losses, val_losses, train_accuracies, val_accuracies = trainer.train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion_cls=criterion_cls,
            criterion_sim=criterion_sim,
            optimizer=optimizer,
            scheduler=scheduler
        )
        
        # Plot training progress
        logger.info("Generating training plots...")
        plot_training_progress(
            train_losses=train_losses,
            val_losses=val_losses,
            train_accuracies=train_accuracies,
            val_accuracies=val_accuracies,
            save_dir=config['output']['plots_dir']
        )
        
        logger.info("Training completed successfully!")
        
    except KeyboardInterrupt:
        logger.info("Training interrupted by user!")
        save_path = os.path.join(config['output']['checkpoints_dir'], 'interrupted_state.pth')
        torch.save({
            'epoch': start_epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'config': config
        }, save_path)
        logger.info(f"Saved interrupted state to {save_path}")
        
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        import traceback
        traceback.print_exc()
        
        if 'model' in locals():
            save_path = os.path.join(config['output']['checkpoints_dir'], 'error_state.pth')
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'config': config
            }, save_path)
            logger.info(f"Saved error state to {save_path}")
        raise

if __name__ == "__main__":
    main()