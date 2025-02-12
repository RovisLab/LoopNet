# evaluate.py
from matplotlib import pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import logging
import os
from pathlib import Path
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import argparse

from data.dataset import CustomDataset
from models.resnet_disk import ResNet18WithDISK
from models.loss import ContrastiveLoss
from utils.config import load_config, get_transforms
from utils.visualization import (
    plot_confusion_matrix,
    plot_feature_distributions,
    save_sample_predictions
)

class Evaluator:
    def __init__(self, config_path: str = "configs/config.yaml"):
        """Initialize evaluator with configuration"""
        self.config = load_config(config_path)
        self.setup_logging()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.logger.info(f"Using device: {self.device}")
        
        # Create output directories
        os.makedirs(self.config['output']['analysis_dir'], exist_ok=True)

    def setup_logging(self):
        """Setup logging configuration"""
        os.makedirs('logs', exist_ok=True)
        logging.basicConfig(
            level=self.config['output']['log_level'],
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('logs/evaluation.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

    def load_model(self, checkpoint_path):
        """Load the trained model"""
        model = ResNet18WithDISK(
            num_classes=self.config['model']['num_classes'],
            device=self.device
        )
        
        if not os.path.exists(checkpoint_path):
            self.logger.error(f"No checkpoint found at {checkpoint_path}")
            raise FileNotFoundError(f"No checkpoint found at {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(self.device)
        model.eval()
        
        self.logger.info(f"Loaded model from {checkpoint_path}")
        return model

    def evaluate(self, model, test_loader):
        """Evaluate the model"""
        results = {
            'predictions': [],
            'features': [],
            'confidences': [],
            'image_paths': []
        }
        
        softmax = nn.Softmax(dim=1)
        
        self.logger.info("Starting evaluation...")
        with torch.no_grad():
            for batch_idx, (inputs, _, paths) in enumerate(test_loader):
                inputs = inputs.to(self.device)
                
                # Get model outputs
                cls_output, sim_output, feat_output = model(inputs)
                probs = softmax(cls_output)
                
                # Store predictions and features
                batch_predictions = probs.argmax(1).cpu().numpy()
                batch_confidences = probs.max(1)[0].cpu().numpy()
                batch_features = feat_output.cpu().numpy()
                
                results['predictions'].extend(batch_predictions)
                results['features'].extend(batch_features)
                results['confidences'].extend(batch_confidences)
                results['image_paths'].extend(paths)
                
                if (batch_idx + 1) % 10 == 0:
                    self.logger.info(f"Processed {batch_idx + 1} batches")
        
        return results
    def save_predictions(self, results, classes):
        """Save detailed predictions to file"""
        output_path = os.path.join(self.config['output']['analysis_dir'], 'predictions.txt')
        
        with open(output_path, 'w') as f:
            f.write("Image Path | Predicted Class | Confidence\n")
            f.write("-" * 60 + "\n")
            
            for path, pred, conf in zip(
                results['image_paths'],
                results['predictions'],
                results['confidences']
            ):
                f.write(f"{path} | {classes[pred]} | {conf:.4f}\n")
        
        self.logger.info(f"Predictions saved to {output_path}")

    def analyze_results(self, results, classes):
        """Analyze evaluation results - modified for unlabeled test data"""
        # Save predictions
        self.save_predictions(results, classes)
        
        # Analyze feature distributions
        feature_stats = plot_feature_distributions(
            results,
            save_dir=self.config['output']['plots_dir']
        )
        
        # Analyze confidence distribution
        plt.figure(figsize=(10, 6))
        plt.hist(results['confidences'], bins=50)
        plt.title('Prediction Confidence Distribution')
        plt.xlabel('Confidence')
        plt.ylabel('Count')
        confidence_plot_path = os.path.join(self.config['output']['plots_dir'], 'confidence_distribution.png')
        plt.savefig(confidence_plot_path)
        plt.close()
        
        # Save class distribution
        pred_counts = np.bincount(results['predictions'], minlength=len(classes))
        distribution_path = os.path.join(self.config['output']['analysis_dir'], 'class_distribution.txt')
        
        with open(distribution_path, 'w') as f:
            f.write("Predicted Class Distribution\n")
            f.write("==========================\n\n")
            
            total_images = len(results['predictions'])
            for i, count in enumerate(pred_counts):
                percentage = (count/total_images)*100
                f.write(f"{classes[i]}: {count} images ({percentage:.2f}%)\n")
                
            f.write("\nConfidence Analysis\n")
            f.write("==================\n")
            f.write(f"Mean confidence: {np.mean(results['confidences']):.4f}\n")
            f.write(f"Std confidence: {np.std(results['confidences']):.4f}\n")
            f.write(f"Min confidence: {np.min(results['confidences']):.4f}\n")
            f.write(f"Max confidence: {np.max(results['confidences']):.4f}\n")
            
            # Per-class confidence analysis
            f.write("\nPer-Class Confidence Analysis\n")
            f.write("===========================\n")
            for i in range(len(classes)):
                mask = np.array(results['predictions']) == i
                if np.any(mask):
                    class_confidences = np.array(results['confidences'])[mask]
                    f.write(f"\n{classes[i]}:\n")
                    f.write(f"  Mean confidence: {np.mean(class_confidences):.4f}\n")
                    f.write(f"  Std confidence: {np.std(class_confidences):.4f}\n")
                    f.write(f"  Count: {np.sum(mask)}\n")
        
        # Save detailed image predictions
        predictions_path = os.path.join(self.config['output']['analysis_dir'], 'image_predictions.txt')
        with open(predictions_path, 'w') as f:
            f.write("Image Predictions\n")
            f.write("================\n\n")
            f.write("Format: image_path | predicted_class | confidence\n\n")
            
            for img_path, pred, conf in zip(
                results['image_paths'],
                results['predictions'],
                results['confidences']
            ):
                f.write(f"{img_path} | {classes[pred]} | {conf:.4f}\n")
        
        self.logger.info(f"Saved predictions analysis to {distribution_path}")
        self.logger.info(f"Saved confidence distribution plot to {confidence_plot_path}")
        self.logger.info(f"Saved detailed predictions to {predictions_path}")
        
        return {
            'feature_stats': feature_stats,
            'prediction_distribution': dict(zip(classes, pred_counts)),
            'confidence_stats': {
                'mean': np.mean(results['confidences']),
                'std': np.std(results['confidences']),
                'min': np.min(results['confidences']),
                'max': np.max(results['confidences'])
            }
        }
    # def analyze_confidence(self, confidences, predictions, true_labels, classes):
    #     """Analyze prediction confidence"""
    #     # Calculate average confidence per class
    #     class_confidences = {}
    #     for class_idx, class_name in enumerate(classes):
    #         mask = predictions == class_idx
    #         if mask.any():
    #             avg_conf = confidences[mask].mean()
    #             class_confidences[class_name] = avg_conf
        
    #     # Save confidence analysis
    #     conf_path = os.path.join(self.config['output']['analysis_dir'], 'confidence_analysis.txt')
    #     with open(conf_path, 'w') as f:
    #         f.write("Confidence Analysis\n")
    #         f.write("===================\n\n")
    #         f.write("Average confidence per class:\n")
    #         for class_name, conf in class_confidences.items():
    #             f.write(f"{class_name}: {conf:.4f}\n")
            
    #         f.write(f"\nOverall average confidence: {confidences.mean():.4f}")
    #         f.write(f"\nConfidence std deviation: {confidences.std():.4f}")

    # def save_detailed_results(self, results, classes):
    #     """Save detailed evaluation results"""
    #     output_path = os.path.join(self.config['output']['analysis_dir'], 'detailed_results.txt')
        
    #     with open(output_path, 'w') as f:
    #         f.write("Detailed Evaluation Results\n")
    #         f.write("==========================\n\n")
            
    #         for idx, (pred, true, conf, path) in enumerate(zip(
    #             results['predictions'],
    #             results['true_labels'],
    #             results['confidences'],
    #             results['image_paths']
    #         )):
    #             f.write(f"Image {idx + 1}: {path}\n")
    #             f.write(f"True class: {classes[true]}\n")
    #             f.write(f"Predicted class: {classes[pred]}\n")
    #             f.write(f"Confidence: {conf:.4f}\n")
    #             f.write("-" * 50 + "\n")

def main():
    parser = argparse.ArgumentParser(description='Model Evaluation')
    parser.add_argument('--config', default='configs/config.yaml', help='Path to config file')
    parser.add_argument('--checkpoint', required=True, help='Path to model checkpoint')
    args = parser.parse_args()

    # Initialize evaluator
    evaluator = Evaluator(args.config)
    
    # Load configuration and setup data
    config = load_config(args.config)
    _, val_transform = get_transforms(config)
    
    # First load training dataset to get classes
    train_dataset = CustomDataset(
        config['data']['train_path'],
        transform=val_transform,
        is_test=False
    )
    
    # Create test dataset and loader
    test_dataset = CustomDataset(
        config['data']['test_path'],
        transform=val_transform,
        is_test=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config['data']['batch_size'],
        shuffle=False,
        num_workers=config['data']['num_workers'],
        pin_memory=config['data']['pin_memory']
    )

    try:
        # Load model
        model = evaluator.load_model(args.checkpoint)
        
        # Run evaluation
        results = evaluator.evaluate(model, test_loader)
        
        # Use training dataset classes for analysis
        analysis = evaluator.analyze_results(results, train_dataset.classes)
        
        # Save sample predictions
        save_sample_predictions(
            model=model,
            test_loader=test_loader,
            device=evaluator.device,
            train_classes=train_dataset.classes,
            num_samples=5,
            save_dir=config['output']['sample_predictions_dir']
        )
        
        evaluator.logger.info("Evaluation completed successfully!")
        
    except Exception as e:
        evaluator.logger.error(f"An error occurred during evaluation: {str(e)}")
        raise

if __name__ == "__main__":
    main()