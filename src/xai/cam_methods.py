from collections.abc import Sequence

import numpy as np
import torch
from torch import nn

from pytorch_grad_cam import LayerCAM, GradCAMPlusPlus
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

def _run_cam(
    cam_class,
    model: nn.Module,
    target_layers: Sequence[nn.Module],
    input_tensor: torch.Tensor,
    target_class: int,
    rgb_float: np.ndarray,
) -> np.ndarray:
    """Run one CAM method and return an RGB heatmap overlay."""
    if input_tensor.ndim != 4 or input_tensor.shape[0] != 1:
        raise ValueError(
            "CAM methods expect one image with shape (1, 3, height, width); "
            f"received {tuple(input_tensor.shape)}."
        )

    targets = [ClassifierOutputTarget(target_class)]
    # BaseCAM is a context manager; exiting it removes forward/backward hooks.
    with cam_class(model=model, target_layers=list(target_layers)) as cam:
        grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0]

    return show_cam_on_image(rgb_float, grayscale_cam, use_rgb=True)

def run_layercam(
    model: nn.Module,
    target_layers: Sequence[nn.Module],
    input_tensor: torch.Tensor,
    target_class: int,
    rgb_float: np.ndarray,
) -> np.ndarray:
    return _run_cam(LayerCAM, model, target_layers, input_tensor, target_class, rgb_float)

def run_gradcam_plusplus(
    model: nn.Module,
    target_layers: Sequence[nn.Module],
    input_tensor: torch.Tensor,
    target_class: int,
    rgb_float: np.ndarray,
) -> np.ndarray:
    return _run_cam(GradCAMPlusPlus, model, target_layers, input_tensor, target_class, rgb_float)
