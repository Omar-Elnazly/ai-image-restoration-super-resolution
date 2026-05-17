"""
SRCNN Training Script — Phase 1

This is the main training loop for SRCNN.

Key steps per epoch:
  1. Load LR/HR patch pairs
  2. Bicubic-upsample LR to HR size
  3. Feed upsampled LR to SRCNN
  4. Compute MSE loss vs HR target
  5. Backpropagate with AMP (mixed precision)
  6. Log metrics, save validation images, save checkpoints

Resume training:
  Set resume=True and provide checkpoint path.
  The optimizer state is restored so training continues correctly.
"""

import os
import sys
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import torchvision.transforms.functional as TF

# Add project root to path so we can import our modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datasets.div2k_dataset import DIV2KDataset
from models.srcnn import build_srcnn
from utils.metrics import calculate_psnr, calculate_ssim, MetricsTracker
from utils.checkpoint import save_checkpoint, load_checkpoint, get_latest_checkpoint
from utils.logger import TensorBoardLogger
from utils.image_utils import save_image_tensor


def train_srcnn(config_path: str = "configs/config.yaml", resume: bool = False):
    """
    Main training function for SRCNN.
    
    Args:
        config_path: Path to config.yaml
        resume:      If True, resume from latest checkpoint
    """

    # ----------------------------------------------------------------
    # 1. Load Configuration
    # ----------------------------------------------------------------
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # ----------------------------------------------------------------
    # 2. Set Seeds for Reproducibility
    # ----------------------------------------------------------------
    seed = config['project']['seed']
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    # ----------------------------------------------------------------
    # 3. Device Setup
    # ----------------------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*50}")
    print(f"  Training SRCNN")
    print(f"  Device: {device}")
    if device.type == 'cuda':
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print(f"{'='*50}\n")

    # ----------------------------------------------------------------
    # 4. Build Datasets
    # ----------------------------------------------------------------
    data_cfg = config['data']
    patch_cfg = config['patches']
    dl_cfg = config['dataloader']

    data_root = data_cfg['root']

    train_dataset = DIV2KDataset(
        hr_dir=os.path.join(data_root, data_cfg['train_hr']),
        lr_dir=os.path.join(data_root, data_cfg['train_lr']),
        hr_patch_size=patch_cfg['hr_patch_size'],
        scale=data_cfg['scale_factor'],
        max_images=data_cfg['max_train_images'],
        augment=True,
        split='train',
    )

    val_dataset = DIV2KDataset(
        hr_dir=os.path.join(data_root, data_cfg['valid_hr']),
        lr_dir=os.path.join(data_root, data_cfg['valid_lr']),
        hr_patch_size=patch_cfg['hr_patch_size'],
        scale=data_cfg['scale_factor'],
        max_images=data_cfg['max_valid_images'],
        augment=False,
        split='val',
    )

    # ----------------------------------------------------------------
    # 5. Build DataLoaders
    # ----------------------------------------------------------------
    # pin_memory=True: pre-allocates tensors in pinned (page-locked) memory
    # This speeds up CPU→GPU transfers significantly

    train_loader = DataLoader(
        train_dataset,
        batch_size=dl_cfg['batch_size'],
        shuffle=True,          # Shuffle for training
        num_workers=dl_cfg['num_workers'],
        pin_memory=dl_cfg['pin_memory'],
        drop_last=True,        # Drop incomplete last batch for stable training
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=dl_cfg['batch_size'],
        shuffle=False,         # Don't shuffle validation
        num_workers=dl_cfg['num_workers'],
        pin_memory=dl_cfg['pin_memory'],
    )

    # ----------------------------------------------------------------
    # 6. Build Model
    # ----------------------------------------------------------------
    model = build_srcnn(config).to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Model parameters: {total_params:,}")

    # ----------------------------------------------------------------
    # 7. Loss Function and Optimizer
    # ----------------------------------------------------------------
    criterion = nn.MSELoss()  # L2 loss standard for SRCNN

    train_cfg = config['training']
    optimizer = optim.Adam(
        model.parameters(),
        lr=train_cfg['learning_rate'],
        weight_decay=train_cfg['weight_decay'],
    )

    # Learning rate scheduler: reduce LR when PSNR plateaus
    # patience=10: wait 10 epochs before reducing
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=10, verbose=True
    )

    # ----------------------------------------------------------------
    # 8. Mixed Precision Setup
    # ----------------------------------------------------------------
    # GradScaler prevents underflow in float16 during backprop
    use_amp = train_cfg['use_amp'] and (device.type == 'cuda')
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    if use_amp:
        print("  Mixed precision (AMP) ENABLED")
    else:
        print("  Mixed precision DISABLED (running in float32)")

    # ----------------------------------------------------------------
    # 9. Paths Setup
    # ----------------------------------------------------------------
    paths = config['paths']
    checkpoint_dir = os.path.join(paths['checkpoints'], 'srcnn')
    val_image_dir = os.path.join(paths['validation_images'], 'srcnn')
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(val_image_dir, exist_ok=True)

    # ----------------------------------------------------------------
    # 10. Resume Training (Optional)
    # ----------------------------------------------------------------
    start_epoch = 0
    best_psnr = 0.0

    if resume:
        latest = get_latest_checkpoint(checkpoint_dir, prefix='srcnn')
        if latest:
            ckpt = load_checkpoint(latest, model, optimizer, scaler, device)
            start_epoch = ckpt['epoch'] + 1
            best_psnr = ckpt['best_psnr']
        else:
            print("  No checkpoint found, starting from scratch.")

    # ----------------------------------------------------------------
    # 11. TensorBoard Logger
    # ----------------------------------------------------------------
    logger = TensorBoardLogger(paths['logs'], experiment_name='srcnn')

    # ----------------------------------------------------------------
    # 12. Training Loop
    # ----------------------------------------------------------------
    scale = data_cfg['scale_factor']

    for epoch in range(start_epoch, train_cfg['epochs']):
        print(f"\n--- Epoch {epoch + 1}/{train_cfg['epochs']} ---")

        # ---- Training Phase ----
        model.train()
        tracker = MetricsTracker()

        progress = tqdm(train_loader, desc="  Training", leave=False)

        for batch_idx, (lr_batch, hr_batch) in enumerate(progress):
            # Move data to GPU
            lr_batch = lr_batch.to(device, non_blocking=True)  # (B, C, lr_H, lr_W)
            hr_batch = hr_batch.to(device, non_blocking=True)  # (B, C, hr_H, hr_W)

            # SRCNN needs bicubic-upscaled input (same spatial size as HR)
            lr_upscaled = torch.nn.functional.interpolate(
                lr_batch,
                scale_factor=scale,
                mode='bicubic',
                align_corners=False,
            )  # (B, C, hr_H, hr_W)

            # Forward pass with AMP context
            with torch.cuda.amp.autocast(enabled=use_amp):
                pred = model(lr_upscaled)
                loss = criterion(pred, hr_batch)

            # Backward pass
            optimizer.zero_grad(set_to_none=True)  # Slightly faster than zero_grad()
            scaler.scale(loss).backward()

            # Gradient clipping: prevents exploding gradients
            if train_cfg['clip_grad_norm'] > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), train_cfg['clip_grad_norm']
                )

            scaler.step(optimizer)
            scaler.update()

            # Track metrics
            batch_psnr = calculate_psnr(pred.detach(), hr_batch)
            tracker.update('loss', loss.item(), lr_batch.size(0))
            tracker.update('psnr', batch_psnr, lr_batch.size(0))

            # Update progress bar
            progress.set_postfix(loss=f"{loss.item():.4f}", psnr=f"{batch_psnr:.2f}")

            # Log to TensorBoard every log_interval batches
            if batch_idx % train_cfg['log_interval'] == 0:
                global_step = epoch * len(train_loader) + batch_idx
                logger.log_scalar('Train/Loss', loss.item(), global_step)
                logger.log_scalar('Train/PSNR', batch_psnr, global_step)

        train_avg = tracker.summary()
        print(f"  Train — Loss: {train_avg['loss']:.4f} | PSNR: {train_avg['psnr']:.2f} dB")

        # ---- Validation Phase ----
        if (epoch + 1) % train_cfg['val_interval'] == 0:
            val_psnr, val_ssim = validate(
                model, val_loader, device, use_amp, scale,
                epoch, val_image_dir, logger
            )

            # Update learning rate scheduler
            scheduler.step(val_psnr)
            logger.log_lr(optimizer, epoch)

            # Save best model
            if val_psnr > best_psnr:
                best_psnr = val_psnr
                best_path = os.path.join(checkpoint_dir, 'srcnn_best.pth')
                save_checkpoint(model, optimizer, epoch, best_psnr, config, best_path, scaler)
                print(f"  *** New best PSNR: {best_psnr:.2f} dB — checkpoint saved ***")

        # Save periodic checkpoint
        if (epoch + 1) % train_cfg['save_interval'] == 0:
            epoch_path = os.path.join(checkpoint_dir, f'srcnn_epoch_{epoch+1:04d}.pth')
            save_checkpoint(model, optimizer, epoch, best_psnr, config, epoch_path, scaler)

    logger.close()
    print(f"\n  Training complete. Best PSNR: {best_psnr:.2f} dB")


def validate(model, val_loader, device, use_amp, scale, epoch, val_image_dir, logger):
    """
    Run validation loop and compute PSNR/SSIM on validation set.
    Also saves a sample of validation images for visual inspection.
    """
    model.eval()
    tracker = MetricsTracker()
    saved_sample = False  # Only save one batch of images

    with torch.no_grad():
        for lr_batch, hr_batch in tqdm(val_loader, desc="  Validation", leave=False):
            lr_batch = lr_batch.to(device, non_blocking=True)
            hr_batch = hr_batch.to(device, non_blocking=True)

            lr_upscaled = torch.nn.functional.interpolate(
                lr_batch, scale_factor=scale, mode='bicubic', align_corners=False
            )

            with torch.cuda.amp.autocast(enabled=use_amp):
                pred = model(lr_upscaled)

            psnr = calculate_psnr(pred, hr_batch)
            ssim = calculate_ssim(pred, hr_batch)
            tracker.update('psnr', psnr, lr_batch.size(0))
            tracker.update('ssim', ssim, lr_batch.size(0))

            # Save validation images for the first batch only
            if not saved_sample:
                for i in range(min(4, pred.size(0))):  # Save up to 4 images
                    save_image_tensor(
                        lr_upscaled[i],
                        os.path.join(val_image_dir, f"epoch_{epoch+1:04d}_img{i}_input.png")
                    )
                    save_image_tensor(
                        pred[i],
                        os.path.join(val_image_dir, f"epoch_{epoch+1:04d}_img{i}_pred.png")
                    )
                    save_image_tensor(
                        hr_batch[i],
                        os.path.join(val_image_dir, f"epoch_{epoch+1:04d}_img{i}_target.png")
                    )
                saved_sample = True

    avg = tracker.summary()
    print(f"  Val   — PSNR: {avg['psnr']:.2f} dB | SSIM: {avg['ssim']:.4f}")

    logger.log_scalar('Val/PSNR', avg['psnr'], epoch)
    logger.log_scalar('Val/SSIM', avg['ssim'], epoch)

    return avg['psnr'], avg['ssim']


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train SRCNN")
    parser.add_argument('--config', type=str, default='configs/config.yaml')
    parser.add_argument('--resume', action='store_true', help='Resume from latest checkpoint')
    args = parser.parse_args()

    train_srcnn(args.config, args.resume)