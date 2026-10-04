import torch
import torch.nn as nn
from torchvision.models import densenet121, DenseNet121_Weights


class DenseNet121Classifier(nn.Module):
    """DenseNet121 model for classification with customizable output classes.
    
    Uses pretrained weights from torchvision. The final classifier layer
    is replaced to match the target number of classes.
    
    Args:
        num_classes: Number of output classes
        pretrained: Whether to load ImageNet pretrained weights
        freeze_features: Whether to freeze the feature extraction layers
    """
    
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
        """Forward pass.
        
        Args:
            x: Input tensor of shape (batch_size, 3, 224, 224)
            
        Returns:
            Logits of shape (batch_size, num_classes)
        """
        return self.model(x)
    
    def unfreeze_features(self):
        """Unfreeze all feature extraction layers for fine-tuning."""
        for param in self.model.features.parameters():
            param.requires_grad = True
    
    def freeze_features(self):
        """Freeze all feature extraction layers."""
        for param in self.model.features.parameters():
            param.requires_grad = False

