from pytorch_grad_cam import LayerCAM, GradCAMPlusPlus
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

def _run_cam(cam_class, model, target_layers, input_tensor, target_class, rgb_float):
    cam = cam_class(model=model, target_layers=target_layers)
    targets = [ClassifierOutputTarget(target_class)]
    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0] 
    return show_cam_on_image(rgb_float, grayscale_cam, use_rgb=True)

def run_layercam(model, target_layers, input_tensor, target_class, rgb_float):
    return _run_cam(LayerCAM, model, target_layers, input_tensor, target_class, rgb_float)

def run_gradcam_plusplus(model, target_layers, input_tensor, target_class, rgb_float):
    return _run_cam(GradCAMPlusPlus, model, target_layers, input_tensor, target_class, rgb_float)