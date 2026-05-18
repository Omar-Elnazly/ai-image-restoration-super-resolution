"""
Denoising Dataset.

For denoising, we don't need LR/HR pairs.
We take HR images and synthetically add noise to create input/target pairs:
  - Input:  Noisy image (HR + noise)
  - Target: Clean image (HR)

This is called "self-supervised" or "synthetic degradation" training.
"""

import os
import random
from typing import Optional, Tuple

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF


class DenoisingDataset(Dataset):
    """
    Dataset for training denoising autoencoder.
    
    Supports:
      - Gaussian noise (simulates sensor noise)
      - JPEG compression artifacts (simulates low-quality images)
      - Mixed degradation (both at once)
    
    Args:
        hr_dir:       Path to clean HR images
        patch_size:   Patch size for training crops
        max_images:   Limit to first N images
        noise_type:   'gaussian', 'jpeg', or 'mixed'
        noise_level:  Standard deviation for Gaussian noise (0.0 to 1.0)
        jpeg_quality: JPEG quality factor (1=worst, 95=best) for JPEG artifacts
        augment:      Apply data augmentation
    """

    def __init__(
        self,
        hr_dir: str,
        patch_size: int = 128,
        max_images: Optional[int] = None,
        noise_type: str = 'gaussian',
        noise_level: float = 0.1,
        jpeg_quality: int = 30,
        augment: bool = True,
    ):
        self.hr_dir = hr_dir
        self.patch_size = patch_size
        self.noise_type = noise_type
        self.noise_level = noise_level
        self.jpeg_quality = jpeg_quality
        self.augment = augment

        # Collect image paths
        self.image_paths = self._collect_images(max_images)
        print(f"  [Denoising Dataset] {len(self.image_paths)} images | noise: {noise_type}")

    def _collect_images(self, max_images: Optional[int]) -> list:
        files = sorted([
            os.path.join(self.hr_dir, f)
            for f in os.listdir(self.hr_dir)
            if f.lower().endswith(('.png', '.jpg', '.jpeg'))
        ])
        if max_images is not None:
            files = files[:max_images]
        return files

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            noisy_tensor: Degraded image tensor (C, H, W), [0, 1]
            clean_tensor: Clean ground truth tensor (C, H, W), [0, 1]
        """
        img = Image.open(self.image_paths[idx]).convert("RGB")

        # Crop a random patch
        patch = self._random_crop(img)

        # Apply augmentation
        if self.augment:
            patch = self._augment(patch)

        # Convert to tensor
        clean_tensor = TF.to_tensor(patch)

        # Add synthetic degradation
        noisy_tensor = self._add_degradation(clean_tensor)

        return noisy_tensor, clean_tensor

    def _random_crop(self, img: Image.Image) -> Image.Image:
        """Crop a random square patch from the image."""
        w, h = img.size
        if w < self.patch_size:
            img = img.resize((self.patch_size, h), Image.BICUBIC)
            w = self.patch_size
        if h < self.patch_size:
            img = img.resize((w, self.patch_size), Image.BICUBIC)
            h = img.size[1]

        x = random.randint(0, w - self.patch_size)
        y = random.randint(0, h - self.patch_size)
        return img.crop((x, y, x + self.patch_size, y + self.patch_size))

    def _augment(self, img: Image.Image) -> Image.Image:
        """Random flip/rotation augmentation."""
        if random.random() > 0.5:
            img = TF.hflip(img)
        if random.random() > 0.5:
            img = TF.vflip(img)
        if random.random() > 0.5:
            img = TF.rotate(img, 90)
        return img

    def _add_degradation(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Add synthetic degradation to a clean image tensor.
        
        Gaussian noise: randomly sampled per pixel, simulates sensor noise.
        JPEG artifacts:  simulate compression artifacts from saving as JPEG.
        """
        if self.noise_type == 'gaussian':
            return self._add_gaussian_noise(tensor)
        elif self.noise_type == 'jpeg':
            return self._add_jpeg_artifacts(tensor)
        elif self.noise_type == 'mixed':
            # Randomly pick one type per sample
            if random.random() > 0.5:
                return self._add_gaussian_noise(tensor)
            else:
                return self._add_jpeg_artifacts(tensor)
        else:
            raise ValueError(f"Unknown noise_type: {self.noise_type}")

    def _add_gaussian_noise(self, tensor: torch.Tensor) -> torch.Tensor:
        """Add zero-mean Gaussian noise with random strength."""
        # Vary noise level slightly per sample for robustness
        level = random.uniform(self.noise_level * 0.5, self.noise_level * 1.5)
        noise = torch.randn_like(tensor) * level
        noisy = (tensor + noise).clamp(0, 1)
        return noisy

    def _add_jpeg_artifacts(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Simulate JPEG compression artifacts.
        We do this by:
          1. Converting tensor back to PIL image
          2. Saving as JPEG with low quality to a buffer
          3. Reloading from buffer
        """
        import io

        # Convert tensor to PIL
        pil_img = TF.to_pil_image(tensor.clamp(0, 1))

        # Vary JPEG quality for robustness
        quality = random.randint(max(5, self.jpeg_quality - 10), self.jpeg_quality + 10)

        # Save to in-memory buffer with JPEG compression
        buffer = io.BytesIO()
        pil_img.save(buffer, format='JPEG', quality=quality)
        buffer.seek(0)

        # Reload the compressed image
        compressed = Image.open(buffer).convert("RGB")
        return TF.to_tensor(compressed)