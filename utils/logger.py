"""
TensorBoard logging wrapper.

TensorBoard lets you visualize training curves, validation images,
and metrics in real time in your browser.

To launch TensorBoard:
    tensorboard --logdir logs/
Then open http://localhost:6006 in your browser.
"""

import os
from torch.utils.tensorboard import SummaryWriter
import torch


class TensorBoardLogger:
    """
    Thin wrapper around SummaryWriter for cleaner logging calls.
    """

    def __init__(self, log_dir: str, experiment_name: str):
        """
        Args:
            log_dir:         Root directory for all logs (e.g., 'logs/')
            experiment_name: Name for this run (e.g., 'srcnn_run1')
        """
        full_log_dir = os.path.join(log_dir, experiment_name)
        os.makedirs(full_log_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir=full_log_dir)
        print(f"  [TensorBoard] Logging to {full_log_dir}")

    def log_scalar(self, tag: str, value: float, step: int):
        """Log a single scalar value (loss, PSNR, SSIM, etc.)."""
        self.writer.add_scalar(tag, value, step)

    def log_scalars(self, tag: str, value_dict: dict, step: int):
        """Log multiple scalars under one tag (e.g., train vs val loss)."""
        self.writer.add_scalars(tag, value_dict, step)

    def log_image(self, tag: str, image_tensor: torch.Tensor, step: int):
        """
        Log an image to TensorBoard.
        
        Args:
            tag:          Label for the image (e.g., 'validation/output')
            image_tensor: Shape (C, H, W) or (B, C, H, W), values [0, 1]
            step:         Current epoch or iteration
        """
        if image_tensor.dim() == 4:
            # Log the first image in the batch
            self.writer.add_images(tag, image_tensor.clamp(0, 1), step)
        else:
            self.writer.add_image(tag, image_tensor.clamp(0, 1), step)

    def log_lr(self, optimizer: torch.optim.Optimizer, step: int):
        """Log current learning rate from optimizer."""
        for i, group in enumerate(optimizer.param_groups):
            self.writer.add_scalar(f'LearningRate/group_{i}', group['lr'], step)

    def close(self):
        """Flush and close the TensorBoard writer."""
        self.writer.close()