# utils/training.py
import torch
import torch.nn as nn
import torch.optim as optim
import logging
import os
from pathlib import Path
from utils.config import load_config
from torch.cuda.amp import GradScaler
from tqdm import tqdm

class Trainer:
    def __init__(self, config_path: str = "configs/config.yaml"):
        """Initialize trainer with configuration"""
        self.config = load_config(config_path)
        self.setup_logging()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.logger.info(f"Using device: {self.device}")
        self.scaler = GradScaler()
        
        # Create output directories
        self.create_output_dirs()

    def setup_logging(self):
        """Setup logging configuration"""
        os.makedirs('logs', exist_ok=True)
        logging.basicConfig(
            level=self.config['output']['log_level'],
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('logs/training.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

    def create_output_dirs(self):
        """Create necessary output directories"""
        for dir_path in [
            self.config['output']['checkpoints_dir'],
            self.config['output']['plots_dir'],
            self.config['output']['analysis_dir'],
            self.config['output']['sample_predictions_dir']
        ]:
            os.makedirs(dir_path, exist_ok=True)

    def train_epoch(self, model, train_loader, criterion_cls, criterion_sim, optimizer):
        """Train for one epoch"""
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        training_config = self.config['training']

        progress_bar = tqdm(train_loader, desc='Training')
        
        for batch_idx, data in enumerate(progress_bar):
            # Handle different data formats (2 or 3 values)
            if len(data) == 3:
                inputs, labels, _ = data  
            else:
                inputs, labels = data

            inputs, labels = inputs.to(self.device), labels.to(self.device)
            optimizer.zero_grad()

            with torch.cuda.amp.autocast():
                cls_output, sim_output, feat_output = model(inputs)
                loss_cls = criterion_cls(cls_output, labels)
                loss_sim = training_config['sim_loss_weight'] * criterion_sim(
                    sim_output, sim_output, torch.ones_like(labels, device=self.device))
                loss = loss_cls + loss_sim

            self.scaler.scale(loss).backward()
            self.scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), training_config['gradient_clip'])
            self.scaler.step(optimizer)
            self.scaler.update()

            running_loss += loss.item()
            _, predicted = cls_output.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            progress_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100. * correct / total:.2f}%'
            })

        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100. * correct / total
        return epoch_loss, epoch_acc

    def validate_model(self, model, val_loader, criterion_cls, criterion_sim):
        """Validate the model"""
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        training_config = self.config['training']

        progress_bar = tqdm(val_loader, desc='Validation')

        with torch.no_grad():
            for data in progress_bar:
                # Handle different data formats
                if len(data) == 3:
                    inputs, labels, _ = data  
                else:
                    inputs, labels = data

                inputs, labels = inputs.to(self.device), labels.to(self.device)
                cls_output, sim_output, feat_output = model(inputs)
                loss_cls = criterion_cls(cls_output, labels)
                loss_sim = training_config['sim_loss_weight'] * criterion_sim(
                    sim_output, sim_output, torch.ones_like(labels, device=self.device))
                loss = loss_cls + loss_sim
                
                val_loss += loss.item()
                _, predicted = cls_output.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

                progress_bar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'acc': f'{100. * correct / total:.2f}%'
                })

        val_loss = val_loss / len(val_loader)
        val_accuracy = 100. * correct / total

        return val_loss, val_accuracy

    def train_model(self, model, train_loader, val_loader, criterion_cls, criterion_sim, optimizer, scheduler):
        """Train the model"""
        model = model.to(self.device)
        criterion_cls = criterion_cls.to(self.device)
        criterion_sim = criterion_sim.to(self.device)
        
        train_losses, val_losses = [], []
        train_accuracies, val_accuracies = [], []
        best_val_acc = 0
        patience_counter = 0
        
        training_config = self.config['training']
        num_epochs = training_config['num_epochs']

        for epoch in range(num_epochs):
            self.logger.info(f"\nEpoch {epoch+1}/{num_epochs}")
            
            # Training phase
            train_loss, train_accuracy = self.train_epoch(
                model, train_loader, criterion_cls, criterion_sim, optimizer
            )
            train_losses.append(train_loss)
            train_accuracies.append(train_accuracy)

            # Validation phase
            val_loss, val_accuracy = self.validate_model(
                model, val_loader, criterion_cls, criterion_sim
            )
            val_losses.append(val_loss)
            val_accuracies.append(val_accuracy)

            scheduler.step(val_loss)

            # Early stopping check
            if val_accuracy > best_val_acc:
                best_val_acc = val_accuracy
                patience_counter = 0
                self.save_checkpoint(
                    epoch=epoch,
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    val_acc=val_accuracy,
                    config=training_config,
                    train_loss=train_loss,
                    val_loss=val_loss,
                    train_acc=train_accuracy
                )
                self.logger.info(f"New best model saved with validation accuracy: {val_accuracy:.2f}%")
            else:
                patience_counter += 1

            if patience_counter >= training_config['early_stopping_patience']:
                self.logger.info(f"Early stopping triggered after {epoch + 1} epochs")
                break

            self.logger.info(
                f"Epoch Summary - "
                f"Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.2f}%, "
                f"Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.2f}%"
            )

        return train_losses, val_losses, train_accuracies, val_accuracies

    def save_checkpoint(self, epoch, model, optimizer, scheduler, val_acc, config,
                       train_loss, val_loss, train_acc):
        """Save model checkpoint"""
        checkpoint_path = os.path.join(
            self.config['output']['checkpoints_dir'],
            'best_model.pth'
        )
        
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'val_acc': val_acc,
            'config': config,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'train_acc': train_acc
        }, checkpoint_path)
        
        self.logger.info(f"Checkpoint saved to {checkpoint_path}")

    def load_checkpoint(self, model, optimizer=None, scheduler=None, checkpoint_path=None):
        """Load model checkpoint"""
        if checkpoint_path is None:
            checkpoint_path = os.path.join(
                self.config['output']['checkpoints_dir'],
                'best_model.pth'
            )

        if not os.path.exists(checkpoint_path):
            self.logger.warning(f"No checkpoint found at {checkpoint_path}")
            return None

        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            model.load_state_dict(checkpoint['model_state_dict'])
            
            if optimizer is not None and 'optimizer_state_dict' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                
            if scheduler is not None and 'scheduler_state_dict' in checkpoint:
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

            self.logger.info(f"Loaded checkpoint from {checkpoint_path}")
            return checkpoint
        except Exception as e:
            self.logger.error(f"Error loading checkpoint: {str(e)}")
            return None