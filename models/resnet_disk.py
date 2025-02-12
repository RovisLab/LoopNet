import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import ResNet18_Weights
from .disk import DISK

class ResNet18WithDISK(nn.Module):
    def __init__(self, num_classes, device=None):
        super(ResNet18WithDISK, self).__init__()
        
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
            
        self.dropout_rate = 0.6
        
        # DISK feature extractor
        self.disk = DISK()
        self.disk.to(self.device)
        
        # Modified ResNet18 backbone
        self.resnet18 = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        for param in list(self.resnet18.parameters())[:4]:
            param.requires_grad = False
            
        self.resnet18 = self.resnet18.to(self.device)
        self.resnet18.fc = nn.Identity()

        # Enhanced projection layers
        self.disk_projection = nn.Sequential(
            nn.Linear(128, 256),
            nn.BatchNorm1d(256, momentum=0.01),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512, momentum=0.01),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate)
        )

        self.fusion = nn.Sequential(
            nn.Linear(1024, 512),
            nn.LayerNorm(512, eps=1e-6),
            nn.ReLU(),
            nn.Dropout(0.5)
        )

        self.fc_classification = nn.Sequential(
            nn.Linear(512, 256),
            nn.LayerNorm(256, eps=1e-6),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

        self.fc_similarity = nn.Sequential(
            nn.Linear(512, 256),
            nn.LayerNorm(256, eps=1e-6),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.LayerNorm(128)
        )
        
        self = self.to(self.device)

    def calculate_similarity(self, feat1, feat2):
        return torch.cosine_similarity(feat1, feat2)

    def forward(self, x):
        x = x.to(self.device)
        
        disk_features = self.disk.forward(x)["descriptors"]
        disk_features = self.disk_projection(disk_features)
        
        resnet_features = self.resnet18(x)
        
        combined_features = torch.cat([resnet_features, disk_features], dim=1)
        fused_features = self.fusion(combined_features)
        
        classification_output = self.fc_classification(fused_features)
        similarity_output = self.fc_similarity(fused_features)
        
        return classification_output, similarity_output, fused_features
