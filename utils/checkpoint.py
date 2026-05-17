"""
Checkpoint management: saving and loading model states.

Checkpoints save:
  - Model weights
  - Optimizer state (so we can resume training correctly)
  - Epoch number
  - Best PSNR seen so far
  - Configuration used

This allows you to stop training at any point and resume later.
"""

import os
import torch


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    best_psnr: float,
    config: dict,
    filepath: str,
    scaler=None,  # GradScaler for mixed precision
):
    """
    Save a training checkpoint to disk.
    
    Args:
        model:     The neural network model
        optimizer: The optimizer (Adam, SGD, etc.)
        epoch:     Current epoch number
        best_psnr: Best PSNR achieved so far
        config:    The configuration dictionary
        filepath:  Where to save the checkpoint (.pth file)
        scaler:    Optional GradScaler for AMP training
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    checkpoint = {
        'epoch': epoch,
        'best_psnr': best_psnr,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'config': config,
    }

    # Save scaler state if using mixed precision
    if scaler is not None:
        checkpoint['scaler_state_dict'] = scaler.state_dict()

    torch.save(checkpoint, filepath)
    print(f"  [Checkpoint] Saved to {filepath}")


def load_checkpoint(
    filepath: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer = None,
    scaler=None,
    device: torch.device = None,
) -> dict:
    """
    Load a checkpoint and restore model (and optionally optimizer) state.
    
    Args:
        filepath:  Path to the .pth checkpoint file
        model:     Model to load weights into
        optimizer: Optional optimizer to restore state
        scaler:    Optional GradScaler to restore state
        device:    Device to map tensors to (cpu or cuda)
    
    Returns:
        The checkpoint dictionary (contains epoch, best_psnr, config)
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Checkpoint not found: {filepath}")

    # map_location ensures checkpoint loads correctly regardless of
    # whether it was saved on GPU and we're loading on CPU (or different GPU)
    map_location = device if device is not None else 'cpu'
    checkpoint = torch.load(filepath, map_location=map_location)

    # Restore model weights
    model.load_state_dict(checkpoint['model_state_dict'])

    # Restore optimizer state if provided
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    # Restore scaler state if provided
    if scaler is not None and 'scaler_state_dict' in checkpoint:
        scaler.load_state_dict(checkpoint['scaler_state_dict'])

    epoch = checkpoint.get('epoch', 0)
    best_psnr = checkpoint.get('best_psnr', 0.0)
    print(f"  [Checkpoint] Loaded from {filepath} (epoch {epoch}, best PSNR {best_psnr:.2f} dB)")

    return checkpoint


def get_latest_checkpoint(checkpoint_dir: str, prefix: str = "") -> str:
    """
    Find the most recently saved checkpoint in a directory.
    
    Args:
        checkpoint_dir: Folder to search
        prefix:         Optional filename prefix filter (e.g., 'srcnn')
    
    Returns:
        Path to the latest checkpoint, or None if none found
    """
    if not os.path.exists(checkpoint_dir):
        return None

    checkpoints = [
        f for f in os.listdir(checkpoint_dir)
        if f.endswith('.pth') and f.startswith(prefix)
    ]

    if not checkpoints:
        return None

    # Sort by modification time — most recent last
    checkpoints.sort(
        key=lambda f: os.path.getmtime(os.path.join(checkpoint_dir, f))
    )

    return os.path.join(checkpoint_dir, checkpoints[-1])