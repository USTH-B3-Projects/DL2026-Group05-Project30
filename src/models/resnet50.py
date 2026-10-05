import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights

class ResNet50Classifier(nn.Module):
    
    def __init__(self, num_classes: int, pretrained: bool = True, freeze_features: bool = False):
        super().__init__()

        # Load pretrained or random initialized ResNet50
        weights = ResNet50_Weights.DEFAULT if pretrained else None
        self.model = resnet50(weights=weights)

        # Freeze backbone if requested
        if freeze_features:
            for param in self.model.parameters():
                param.requires_grad = False

        # Replace classifier
        num_features = self.model.fc.in_features
        self.model.fc = nn.Linear(num_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def freeze_features(self):
        # Freeze all feature extraction layers
        for name, param in self.model.named_parameters():
            if not name.startswith("fc."):
                param.requires_grad = False

    def unfreeze_features(self):
        # Unfreeze all feature extraction layers for fine-tuning
        for name, param in self.model.named_parameters():
            if not name.startswith("fc."):
                param.requires_grad = True
