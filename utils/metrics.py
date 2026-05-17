"""
Image quality metrics: PSNR and SSIM.

PSNR (Peak Signal-to-Noise Ratio):
  - Measures pixel-level accuracy
  - Higher is better (dB)
  - Typical range for SR: 28-38 dB

SSIM (Structural Similarity Index):
  - Measures perceptual similarity (luminance, contrast, structure)
  - Higher is better, max = 1.0
  - More aligned with human perception than PSNR
"""

import torch
import torch.nn.functional as F
import numpy as np
from skimage.metrics import structural_similarity as sk_ssim


def calculate_psnr(pred: torch.Tensor, target: torch.Tensor, max_val: float = 1.0) -> float:
    """
    Calculate PSNR between predicted and target tensors.
    
    Args:
        pred:    Predicted image tensor, shape (B, C, H, W) or (C, H, W), range [0, 1]
        target:  Ground truth image tensor, same shape
        max_val: Maximum pixel value (1.0 for normalized images)
    
    Returns:
        PSNR value in decibels (float)
    """
    # Clamp to valid range to avoid log(0) issues
    pred = pred.clamp(0, max_val)
    target = target.clamp(0, max_val)

    # Mean Squared Error
    mse = F.mse_loss(pred, target)

    if mse == 0:
        return float('inf')  # Perfect prediction

    # PSNR formula: 10 * log10(MAX^2 / MSE)
    psnr = 10 * torch.log10((max_val ** 2) / mse)
    return psnr.item()


def calculate_ssim(pred: torch.Tensor, target: torch.Tensor) -> float:
    """
    Calculate mean SSIM between predicted and target tensors.
    
    We use scikit-image's SSIM which is the standard implementation.
    This operates on numpy arrays, so we convert from tensors.
    
    Args:
        pred:   Predicted image tensor, shape (B, C, H, W) or (C, H, W), range [0, 1]
        target: Ground truth image tensor, same shape
    
    Returns:
        Mean SSIM value (float) in range [-1, 1], higher is better
    """
    # Ensure we have a batch dimension
    if pred.dim() == 3:
        pred = pred.unsqueeze(0)
        target = target.unsqueeze(0)

    batch_size = pred.shape[0]
    ssim_values = []

    for i in range(batch_size):
        # Convert to numpy (H, W, C) format
        pred_np = pred[i].clamp(0, 1).permute(1, 2, 0).cpu().numpy()
        target_np = target[i].clamp(0, 1).permute(1, 2, 0).cpu().numpy()

        # Calculate SSIM — channel_axis=2 means color channels are last
        ssim_val = sk_ssim(
            pred_np,
            target_np,
            data_range=1.0,
            channel_axis=2,
            win_size=7  # Must be odd and <= min(H, W)
        )
        ssim_values.append(ssim_val)

    return float(np.mean(ssim_values))


class MetricsTracker:
    """
    Tracks running average of metrics during training/validation.
    
    Usage:
        tracker = MetricsTracker()
        for batch in dataloader:
            psnr = calculate_psnr(pred, target)
            tracker.update('psnr', psnr)
        print(tracker.average('psnr'))
        tracker.reset()
    """

    def __init__(self):
        self._sums = {}
        self._counts = {}

    def update(self, name: str, value: float, count: int = 1):
        """Add a new value to the running average."""
        if name not in self._sums:
            self._sums[name] = 0.0
            self._counts[name] = 0
        self._sums[name] += value * count
        self._counts[name] += count

    def average(self, name: str) -> float:
        """Return the running average for a metric."""
        if self._counts.get(name, 0) == 0:
            return 0.0
        return self._sums[name] / self._counts[name]

    def reset(self):
        """Reset all accumulators (call at start of each epoch)."""
        self._sums.clear()
        self._counts.clear()

    def summary(self) -> dict:
        """Return all averages as a dictionary."""
        return {name: self.average(name) for name in self._sums}