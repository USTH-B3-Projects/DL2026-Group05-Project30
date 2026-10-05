import torch
import torch.nn as nn
from torchvision.models import densenet121, DenseNet121_Weights

class DenseNet121Classifier(nn.Module):
    
    def __init__(self, num_classes: int, pretrained: bool = True, freeze_features: bool = False):
        super().__init__()
        
        # Load pretrained or random-initialized model
        weights = DenseNet121_Weights.DEFAULT if pretrained else None
        self.model = densenet121(weights=weights)
        
        # Freeze feature extraction layers if requested
        if freeze_features:
            for param in self.model.features.parameters():
                param.requires_grad = False
        
        # Replace the classifier layer
        # DenseNet121 classifier input features: 1024
        num_features = self.model.classifier.in_features
        self.model.classifier = nn.Linear(num_features, num_classes)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
    
    def freeze_features(self):
        # Freeze all feature extraction layers.
        for param in self.model.features.parameters():
            param.requires_grad = False

    def unfreeze_features(self):
        # Unfreeze all feature extraction layers for fine-tuning.
        for param in self.model.features.parameters():
            param.requires_grad = True
            