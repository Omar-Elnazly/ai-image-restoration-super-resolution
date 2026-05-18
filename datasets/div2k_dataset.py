"""
DIV2K Dataset Loader for Super-Resolution.

DIV2K is a widely used benchmark dataset for image super-resolution.
It contains 800 training and 100 validation high-resolution images.

For x2 super-resolution:
  - Input:  LR image (bicubic downsampled by factor 2)
  - Target: HR image (original high resolution)

We use patch-based training to:
  1. Reduce VRAM usage (small patches fit easily)
  2. Increase effective dataset size (many patches per image)
  3. Allow larger batch sizes for stable training
"""

import os
import random
from typing import Optional, Tuple

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF


class DIV2KDataset(Dataset):
    """
    PyTorch Dataset for DIV2K super-resolution.
    
    Args:
        hr_dir:        Path to directory containing HR images
        lr_dir:        Path to directory containing LR images (x2)
        hr_patch_size: Size of HR patches to crop (e.g., 128 → LR is 64)
        scale:         Scale factor (2 for x2 SR)
        max_images:    Limit dataset to first N images (None = use all)
        augment:       Whether to apply data augmentation (only for training)
        split:         'train' or 'val' — affects augmentation behavior
    """

    def __init__(
        self,
        hr_dir: str,
        lr_dir: str,
        hr_patch_size: int = 128,
        scale: int = 2,
        max_images: Optional[int] = None,
        augment: bool = True,
        split: str = 'train',
    ):
        self.hr_dir = hr_dir
        self.lr_dir = lr_dir
        self.hr_patch_size = hr_patch_size
        self.lr_patch_size = hr_patch_size // scale  # e.g., 128//2 = 64
        self.scale = scale
        self.augment = augment and (split == 'train')

        # Collect all image pairs
        self.image_pairs = self._collect_image_pairs(max_images)

        print(f"  [Dataset] {split.upper()} — {len(self.image_pairs)} images loaded from {hr_dir}")

    def _collect_image_pairs(self, max_images: Optional[int]) -> list:
        """
        Match HR and LR image files.
        
        DIV2K naming convention:
          HR: 0001.png, 0002.png, ...
          LR: 0001x2.png, 0002x2.png, ...
        """
        hr_files = sorted([
            f for f in os.listdir(self.hr_dir)
            if f.lower().endswith(('.png', '.jpg', '.jpeg'))
        ])

        if max_images is not None:
            hr_files = hr_files[:max_images]

        pairs = []
        for hr_file in hr_files:
            hr_path = os.path.join(self.hr_dir, hr_file)

            # Construct the expected LR filename
            # e.g., 0001.png → 0001x2.png
            name, ext = os.path.splitext(hr_file)
            if "celeba" in self.hr_dir.lower():
                lr_file = hr_file
            else:
                lr_file = f"{name}x{self.scale}{ext}"

            lr_path = os.path.join(self.lr_dir, lr_file)

            if os.path.exists(lr_path):
                pairs.append((hr_path, lr_path))
            else:
                print(f"  [Warning] LR file not found for {hr_file}, skipping.")

        return pairs

    def __len__(self) -> int:
        return len(self.image_pairs)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Load one HR/LR image pair, crop a random patch, and return tensors.
        
        Returns:
            lr_patch: Low-resolution patch tensor (C, lr_size, lr_size)
            hr_patch: High-resolution patch tensor (C, hr_size, hr_size)
        """
        hr_path, lr_path = self.image_pairs[idx]

        # Load images as PIL (RGB)
        hr_img = Image.open(hr_path).convert("RGB")
        lr_img = Image.open(lr_path).convert("RGB")

        # Extract a random patch from the LR image, then get the corresponding HR patch
        lr_patch, hr_patch = self._random_crop(lr_img, hr_img)

        # Apply augmentation (random flips and rotation) during training
        if self.augment:
            lr_patch, hr_patch = self._augment(lr_patch, hr_patch)

        # Convert to tensors in [0, 1]
        lr_tensor = TF.to_tensor(lr_patch)  # (C, H, W), float32
        hr_tensor = TF.to_tensor(hr_patch)

        return lr_tensor, hr_tensor

    def _random_crop(self, lr_img: Image.Image, hr_img: Image.Image) -> Tuple[Image.Image, Image.Image]:
        """
        Crop a random patch from LR, then crop the corresponding HR patch.
        
        We crop from LR first (smaller image), then scale coordinates by
        `scale` to get the matching HR region.
        """
        lr_w, lr_h = lr_img.size  # PIL size is (width, height)

        # Ensure image is large enough for a patch
        if lr_w < self.lr_patch_size or lr_h < self.lr_patch_size:
            # If too small, resize LR up and HR accordingly
            # This is a fallback for very small images
            lr_img = lr_img.resize(
                (max(lr_w, self.lr_patch_size), max(lr_h, self.lr_patch_size)),
                Image.BICUBIC
            )
            lr_w, lr_h = lr_img.size

        # Random top-left corner for LR patch
        lr_x = random.randint(0, lr_w - self.lr_patch_size)
        lr_y = random.randint(0, lr_h - self.lr_patch_size)

        # Crop LR patch
        lr_patch = lr_img.crop((lr_x, lr_y, lr_x + self.lr_patch_size, lr_y + self.lr_patch_size))

        # Corresponding HR patch (multiply coordinates by scale factor)
        hr_x = lr_x * self.scale
        hr_y = lr_y * self.scale
        hr_patch = hr_img.crop((hr_x, hr_y, hr_x + self.hr_patch_size, hr_y + self.hr_patch_size))

        return lr_patch, hr_patch

    def _augment(self, lr: Image.Image, hr: Image.Image) -> Tuple[Image.Image, Image.Image]:
        """
        Apply the same random augmentation to both LR and HR patches.
        
        We must apply IDENTICAL transforms to both patches, otherwise
        the model would learn from mismatched pairs.
        
        Augmentations used:
          - Horizontal flip (50% probability)
          - Vertical flip (50% probability)  
          - 90-degree rotation (50% probability)
        """
        # Random horizontal flip
        if random.random() > 0.5:
            lr = TF.hflip(lr)
            hr = TF.hflip(hr)

        # Random vertical flip
        if random.random() > 0.5:
            lr = TF.vflip(lr)
            hr = TF.vflip(hr)

        # Random 90-degree rotation
        if random.random() > 0.5:
            lr = TF.rotate(lr, 90)
            hr = TF.rotate(hr, 90)

        return lr, hr