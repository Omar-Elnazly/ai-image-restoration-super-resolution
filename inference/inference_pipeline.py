"""
Inference Pipeline

Loads a trained model and runs super-resolution or denoising on a full image.

For inference on full images (not patches):
  - The entire image is processed at once
  - For very large images, we pad to a multiple of scale factor
  - SRCNN needs bicubic upsampling as preprocessing step
  - SRResNet/SRGAN take raw LR directly

Usage:
    pipeline = InferencePipeline(model_type='srcnn', checkpoint_path='checkpoints/srcnn/srcnn_best.pth')
    result = pipeline.run('my_image.jpg')
"""

import os
import sys
import torch
import yaml
from PIL import Image
import torchvision.transforms.functional as TF
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.srcnn import build_srcnn
from models.autoencoder import DenoisingAutoencoder
from models.srresnet import SRResNet
from models.srgan import SRGANDiscriminator
from utils.checkpoint import load_checkpoint
from utils.image_utils import save_image_tensor, pad_image_to_multiple, crop_to_original
from utils.metrics import calculate_psnr, calculate_ssim


class InferencePipeline:
    """
    Unified inference pipeline for all model types.
    
    Args:
        model_type:      'srcnn', 'denoising', 'srresnet', or 'srgan'
        checkpoint_path: Path to trained model checkpoint (.pth file)
        config_path:     Path to config.yaml
        device:          'cuda', 'cpu', or 'auto' (auto-detects)
    """

    def __init__(
        self,
        model_type: str,
        checkpoint_path: str,
        config_path: str = "configs/config.yaml",
        device: str = 'auto',
    ):
        self.model_type = model_type.lower()

        # Load config
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        # Setup device
        if device == 'auto':
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.scale = self.config['data']['scale_factor']

        # Build and load model
        self.model = self._build_model()
        load_checkpoint(checkpoint_path, self.model, device=self.device)
        self.model.eval()

        print(f"  [Inference] Model: {model_type} | Device: {self.device}")

    def _build_model(self) -> torch.nn.Module:
        """Build the correct model based on model_type."""
        if self.model_type == 'srcnn':
            return build_srcnn(self.config).to(self.device)

        elif self.model_type == 'denoising':
            return DenoisingAutoencoder(in_channels=3, base_filters=32).to(self.device)

        elif self.model_type in ['srresnet', 'srgan']:
            return SRResNet(
                scale_factor=self.scale,
                num_channels=3,
                num_filters=64,
                num_res_blocks=16,
            ).to(self.device)

        else:
            raise ValueError(f"Unknown model_type: {self.model_type}. "
                             f"Choose from: srcnn, denoising, srresnet, srgan")

    @torch.no_grad()
    def run(
        self,
        input_image,  # str path or PIL Image
        target_image=None,  # Optional: for PSNR/SSIM calculation
        save_path: str = None,
    ) -> dict:
        """
        Run inference on a single image.
        
        Args:
            input_image:  Path to input image (LR for SR, noisy for denoising)
                         OR a PIL Image object
            target_image: Optional HR/clean reference image for metric calculation
            save_path:   If provided, save result to this path
        
        Returns:
            Dictionary with keys:
              - 'output': PIL Image of the result
              - 'input': PIL Image of the input
              - 'psnr': float (if target provided)
              - 'ssim': float (if target provided)
        """

        # ---- Load Input ----
        if isinstance(input_image, str):
            input_pil = Image.open(input_image).convert("RGB")
        elif isinstance(input_image, Image.Image):
            input_pil = input_image.convert("RGB")
        else:
            raise TypeError("input_image must be a file path (str) or PIL Image")

        # ---- Preprocess ----
        input_tensor = TF.to_tensor(input_pil).to(self.device)  # (C, H, W) [0, 1]

        # Pad to multiple of 4 to handle pooling layers (important for autoencoder)
        input_padded, original_size = pad_image_to_multiple(input_tensor, multiple=8)
        input_padded = input_padded.unsqueeze(0)  # Add batch dim: (1, C, H, W)

        # ---- Model-specific preprocessing ----
        if self.model_type == 'srcnn':
            # SRCNN needs bicubic upsampled input
            model_input = torch.nn.functional.interpolate(
                input_padded, scale_factor=self.scale, mode='bicubic', align_corners=False
            )
        else:
            model_input = input_padded

        # ---- Run Inference ----
        with torch.cuda.amp.autocast(enabled=(self.device.type == 'cuda')):
            output = self.model(model_input)

        output = output.squeeze(0)  # Remove batch dim: (C, H, W)

        # ---- Crop padding ----
        if self.model_type == 'srcnn':
            # Output is same size as bicubic input — crop to HR size
            hr_size = (original_size[0] * self.scale, original_size[1] * self.scale)
            output = crop_to_original(output, hr_size)
        elif self.model_type in ['srresnet', 'srgan']:
            # Output is scale * original size
            hr_size = (original_size[0] * self.scale, original_size[1] * self.scale)
            output = crop_to_original(output, hr_size)
        else:
            # Denoising: same size as input
            output = crop_to_original(output, original_size)

        output = output.clamp(0, 1)

        # ---- Convert to PIL ----
        output_pil = TF.to_pil_image(output.cpu())

        # ---- Save if requested ----
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True) if os.path.dirname(save_path) else None
            output_pil.save(save_path)
            print(f"  [Inference] Saved result to: {save_path}")

        # ---- Compute metrics if target provided ----
        result = {
            'output': output_pil,
            'input': input_pil,
        }

        if target_image is not None:
            if isinstance(target_image, str):
                target_pil = Image.open(target_image).convert("RGB")
            else:
                target_pil = target_image.convert("RGB")

            target_tensor = TF.to_tensor(target_pil).to(self.device)

            # Resize target to match output if sizes differ slightly
            if target_tensor.shape[-2:] != output.shape[-2:]:
                target_tensor = torch.nn.functional.interpolate(
                    target_tensor.unsqueeze(0),
                    size=output.shape[-2:],
                    mode='bilinear',
                    align_corners=False
                ).squeeze(0)

            result['psnr'] = calculate_psnr(output, target_tensor)
            result['ssim'] = calculate_ssim(output, target_tensor)
            print(f"  [Inference] PSNR: {result['psnr']:.2f} dB | SSIM: {result['ssim']:.4f}")

        return result


def batch_inference(
    pipeline: InferencePipeline,
    input_dir: str,
    output_dir: str,
    extensions: tuple = ('.png', '.jpg', '.jpeg', '.bmp'),
):
    """
    Run inference on all images in a directory.
    
    Args:
        pipeline:   Initialized InferencePipeline
        input_dir:  Directory containing input images
        output_dir: Directory to save output images
        extensions: File extensions to process
    """
    import glob
    from tqdm import tqdm

    os.makedirs(output_dir, exist_ok=True)

    image_files = []
    for ext in extensions:
        image_files.extend(glob.glob(os.path.join(input_dir, f'*{ext}')))
        image_files.extend(glob.glob(os.path.join(input_dir, f'*{ext.upper()}')))

    if not image_files:
        print(f"No images found in {input_dir}")
        return

    print(f"  Processing {len(image_files)} images...")

    for img_path in tqdm(image_files):
        filename = os.path.basename(img_path)
        save_path = os.path.join(output_dir, filename)
        pipeline.run(img_path, save_path=save_path)

    print(f"  Done. Results saved to {output_dir}")