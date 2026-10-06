import numpy as np
from captum.attr import Occlusion
from pytorch_grad_cam.utils.image import show_cam_on_image

def run_occlusion(model, input_tensor, target_class, rgb_float, sliding_window=(3, 15, 15), strides=(3, 8, 8)):
    occlusion = Occlusion(model)
    attributions = occlusion.attribute(
        input_tensor,
        target=target_class,
        sliding_window_shapes=sliding_window,
        strides=strides,
        baselines=0,
    )

    attr_map = attributions.squeeze().detach().cpu().numpy()  
    attr_map = np.abs(attr_map).sum(axis=0)                    
    attr_map = (attr_map - attr_map.min()) / (attr_map.max() - attr_map.min() + 1e-8)

    return show_cam_on_image(rgb_float, attr_map, use_rgb=True)