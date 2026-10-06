import numpy as np
import torch
from torch import nn

from captum.attr import Occlusion
from pytorch_grad_cam.utils.image import show_cam_on_image

def run_occlusion(
    model: nn.Module,
    input_tensor: torch.Tensor,
    target_class: int,
    rgb_float: np.ndarray,
    sliding_window: tuple[int, int, int] = (3, 15, 15),
    strides: tuple[int, int, int] = (3, 8, 8),
    perturbations_per_eval: int | None = None,
) -> np.ndarray:
    """Return an Occlusion overlay for one image and one target class.

    On CUDA, the default evaluates 16 perturbations together to speed up
    Occlusion. On CPU it evaluates one at a time to limit memory use. Pass an
    explicit value when a different speed / memory trade-off is required.
    """
    if input_tensor.ndim != 4 or input_tensor.shape[0] != 1:
        raise ValueError(
            "Occlusion expects one image with shape (1, 3, height, width); "
            f"received {tuple(input_tensor.shape)}."
        )
    if perturbations_per_eval is None:
        perturbations_per_eval = 16 if input_tensor.is_cuda else 1
    if perturbations_per_eval < 1:
        raise ValueError("perturbations_per_eval must be at least 1.")

    occlusion = Occlusion(model)
    attributions = occlusion.attribute(
        input_tensor,
        target=target_class,
        sliding_window_shapes=sliding_window,
        strides=strides,
        baselines=0,
        perturbations_per_eval=perturbations_per_eval,
    )

    attr_map = attributions.squeeze(0).detach().float().cpu().numpy()
    attr_map = np.abs(attr_map).sum(axis=0)
    attr_map = (attr_map - attr_map.min()) / (attr_map.max() - attr_map.min() + 1e-8)

    return show_cam_on_image(rgb_float, attr_map, use_rgb=True)
