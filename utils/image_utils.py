"""
Image utility functions used throughout the project.
Handles loading, saving, converting, and preprocessing images.

NOTE: OpenCV is imported optionally. All training code uses PIL only.
cv2 is only needed if you explicitly call load_image_cv2().
"""

import os
import numpy as np
from PIL import Image
import torch
import torchvision.transforms.functional as TF

# OpenCV is optional — PIL handles everything needed for training
try:
    import cv2
    OPENCV_AVAILABLE = True
except ImportError:
    OPENCV_AVAILABLE = False


def load_image_pil(path: str) -> Image.Image:
    """
    Load an image using Pillow and convert to RGB.
    This is the primary loader used everywhere in training.
    """
    return Image.open(path).convert("RGB")


def load_image_cv2(path: str) -> np.ndarray:
    """
    Load an image using OpenCV in BGR format, then convert to RGB.
    Only use this if you need OpenCV-specific processing.
    """
    if not OPENCV_AVAILABLE:
        raise ImportError(
            "OpenCV is not available on this system.\n"
            "Use load_image_pil() instead, or install opencv-python-headless."
        )
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Could not load image: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def save_image_tensor(tensor: torch.Tensor, path: str):
    """
    Save a PyTorch tensor as an image file.

    Args:
        tensor: Shape (C, H, W) or (1, C, H, W), values in [0, 1]
        path:   Output file path (e.g., 'outputs/result.png')
    """
    if tensor.dim() == 4:
        tensor = tensor.squeeze(0)

    tensor = tensor.clamp(0, 1)
    img = TF.to_pil_image(tensor.cpu())

    # Create parent directory if it doesn't exist
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)

    img.save(path)


def tensor_to_numpy(tensor: torch.Tensor) -> np.ndarray:
    """
    Convert a (C, H, W) tensor in [0, 1] to (H, W, C) uint8 numpy array.
    """
    tensor = tensor.clamp(0, 1)
    img = tensor.permute(1, 2, 0).cpu().numpy()
    return (img * 255).astype(np.uint8)


def numpy_to_tensor(img: np.ndarray) -> torch.Tensor:
    """
    Convert (H, W, C) uint8 numpy array [0, 255] to (C, H, W) float tensor [0, 1].
    """
    img = img.astype(np.float32) / 255.0
    return torch.from_numpy(img).permute(2, 0, 1)


def pad_image_to_multiple(img: torch.Tensor, multiple: int = 4) -> tuple:
    """
    Pad image so height and width are divisible by multiple.
    Needed for models with pooling layers (autoencoder).

    Returns:
        padded_img:    Padded tensor
        original_size: (H, W) before padding, used for cropping after inference
    """
    _, h, w = img.shape
    pad_h = (multiple - h % multiple) % multiple
    pad_w = (multiple - w % multiple) % multiple
    padded = torch.nn.functional.pad(img, (0, pad_w, 0, pad_h), mode='reflect')
    return padded, (h, w)


def crop_to_original(img: torch.Tensor, original_size: tuple) -> torch.Tensor:
    """Remove padding added by pad_image_to_multiple."""
    h, w = original_size
    return img[:, :h, :w]