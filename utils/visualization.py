import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from PIL import Image
import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix
import logging

logger = logging.getLogger(__name__)

def plot_training_progress(train_losses, val_losses, train_accuracies, val_accuracies, save_dir='./plots'):
    """Plot training progress"""
    os.makedirs(save_dir, exist_ok=True)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    epochs = range(1, len(train_losses) + 1)
    
    # Loss plot
    ax1.plot(epochs, train_losses, 'b-', label='Training')
    ax1.plot(epochs, val_losses, 'r-', label='Validation')
    ax1.set_title('Training and Validation Loss')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Accuracy plot
    ax2.plot(epochs, train_accuracies, 'b-', label='Training')
    ax2.plot(epochs, val_accuracies, 'r-', label='Validation')
    ax2.set_title('Training and Validation Accuracy')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Accuracy (%)')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    save_path = os.path.join(save_dir, 'training_progress.png')
    plt.savefig(save_path)
    plt.close()
    logger.info(f"Training progress plot saved to {save_path}")

def plot_feature_distributions(results, save_dir='./plots'):
    """Plot feature distributions"""
    os.makedirs(save_dir, exist_ok=True)
    features = np.array(results['features'])
    feature_means = np.mean(features, axis=0)
    feature_stds = np.std(features, axis=0)
    
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.hist(feature_means, bins=30)
    plt.title('Feature Means Distribution')
    plt.xlabel('Mean Value')
    plt.ylabel('Count')
    
    plt.subplot(1, 2, 2)
    plt.hist(feature_stds, bins=30)
    plt.title('Feature Standard Deviations')
    plt.xlabel('Standard Deviation')
    plt.ylabel('Count')
    
    plt.tight_layout()
    save_path = os.path.join(save_dir, 'feature_distributions.png')
    plt.savefig(save_path)
    plt.close()
    
    logger.info(f"Feature distributions plot saved to {save_path}")
    return {
        'means': feature_means,
        'stds': feature_stds
    }

def plot_confusion_matrix(results, class_names, save_dir='./plots'):
    """Plot confusion matrix"""
    os.makedirs(save_dir, exist_ok=True)
    cm = confusion_matrix(results['true_labels'], results['predictions'])
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names if class_names else [],
                yticklabels=class_names if class_names else [])
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    
    save_path = os.path.join(save_dir, 'confusion_matrix.png')
    plt.savefig(save_path)
    plt.close()
    
    return cm

def save_sample_predictions(model, test_loader, device, train_classes, num_samples=5, save_dir='sample_predictions'):
    """Save sample predictions with visualizations"""
    os.makedirs(save_dir, exist_ok=True)
    model.eval()
    softmax = nn.Softmax(dim=1)
    
    with torch.no_grad():
        for i, (inputs, _, paths) in enumerate(test_loader):
            if i >= num_samples:
                break
                
            inputs = inputs.to(device)
            cls_output, _, _ = model(inputs)
            probs = softmax(cls_output)
            
            for j in range(len(inputs)):
                img_path = paths[j]
                img = Image.open(img_path).convert('RGB')
                
                # Get top 3 predictions
                top3_probs, top3_classes = torch.topk(probs[j], 3)
                
                # Create figure with image and predictions
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
                
                # Plot image
                ax1.imshow(img)
                ax1.axis('off')
                ax1.set_title('Input Image')
                
                # Plot prediction probabilities
                bars = ax2.bar(range(3), top3_probs.cpu().numpy())
                ax2.set_xticks(range(3))
                ax2.set_xticklabels([train_classes[idx] for idx in top3_classes.cpu().numpy()], 
                                  rotation=45)
                ax2.set_title('Top 3 Predictions')
                ax2.set_ylim(0, 1)
                
                # Add probability values on top of bars
                for idx, rect in enumerate(bars):
                    height = rect.get_height()
                    ax2.text(rect.get_x() + rect.get_width()/2., height,
                            f'{top3_probs[idx]:.2f}',
                            ha='center', va='bottom')
                
                plt.tight_layout()
                save_path = os.path.join(save_dir, f'sample_{i}_{j}.png')
                plt.savefig(save_path)
                plt.close()
                
                # Save prediction details
                details_path = os.path.join(save_dir, f'sample_{i}_{j}_details.txt')
                with open(details_path, 'w') as f:
                    f.write(f"Image: {img_path}\n")
                    f.write("\nTop 3 predictions:\n")
                    for k in range(3):
                        class_idx = top3_classes[k].item()
                        class_name = train_classes[class_idx]
                        prob = top3_probs[k].item()
                        f.write(f"{k+1}. {class_name}: {prob:.4f}\n")
                        
    logger.info(f"Saved {num_samples} sample predictions to {save_dir}")
