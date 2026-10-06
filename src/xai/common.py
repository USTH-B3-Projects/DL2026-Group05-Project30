from pathlib import Path

import numpy as np
import torch

from src.training.config import DEVICE, CHECKPOINT_DIR, NUM_CLASSES
from src.models.resnet50 import ResNet50Classifier

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
IMAGENET_STD = np.array([0.229, 0.224, 0.225])

def load_model(checkpoint_name: str):
    ckpt_path = Path(CHECKPOINT_DIR) / checkpoint_name

    if not ckpt_path.is_file():
        raise FileNotFoundError(
            f"Checkpoint not found: {ckpt_path}. "
            "Download or train the checkpoint before running XAI."
        )

    model = ResNet50Classifier(
        num_classes=NUM_CLASSES,
        pretrained=False,
        freeze_features=False
    ).to(DEVICE)
    state_dict = torch.load(ckpt_path, map_location=DEVICE, weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()

    target_layers = [model.model.layer4[-1]]
    return model, target_layers

def denormalize(tensor_img: torch.Tensor) -> np.ndarray:
    img = tensor_img.detach().cpu().numpy().transpose(1, 2, 0)
    img = img * IMAGENET_STD + IMAGENET_MEAN
    return np.clip(img, 0.0, 1.0).astype(np.float32)
