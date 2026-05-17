"""
Denoising Autoencoder Training Script — Phase 2

Very similar structure to train_srcnn.py but uses DenoisingDataset
and DenoisingAutoencoder.

Key difference: no bicubic upsampling needed — input/output are same size.
"""

import os
import sys
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datasets.denoising_dataset import DenoisingDataset
from models.autoencoder import DenoisingAutoencoder
from utils.metrics import calculate_psnr, calculate_ssim, MetricsTracker
from utils.checkpoint import save_checkpoint, load_checkpoint, get_latest_checkpoint
from utils.logger import TensorBoardLogger
from utils.image_utils import save_image_tensor


def train_denoising(config_path: str = "configs/config.yaml", resume: bool = False):

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    seed = config['project']['seed']
    torch.manual_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*50}")
    print(f"  Training Denoising Autoencoder")
    print(f"  Device: {device}")
    print(f"{'='*50}\n")

    data_cfg = config['data']
    patch_cfg = config['patches']
    dl_cfg = config['dataloader']
    train_cfg = config['training']
    paths = config['paths']

    data_root = data_cfg['root']

    # Note: denoising only needs HR images — no LR needed
    train_dataset = DenoisingDataset(
        hr_dir=os.path.join(data_root, data_cfg['train_hr']),
        patch_size=patch_cfg['hr_patch_size'],
        max_images=data_cfg['max_train_images'],
        noise_type='mixed',
        noise_level=0.1,
        jpeg_quality=30,
        augment=True,
    )

    val_dataset = DenoisingDataset(
        hr_dir=os.path.join(data_root, data_cfg['valid_hr']),
        patch_size=patch_cfg['hr_patch_size'],
        max_images=data_cfg['max_valid_images'],
        noise_type='mixed',
        noise_level=0.1,
        jpeg_quality=30,
        augment=False,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=dl_cfg['batch_size'],
        shuffle=True,
        num_workers=dl_cfg['num_workers'],
        pin_memory=dl_cfg['pin_memory'],
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=dl_cfg['batch_size'],
        shuffle=False,
        num_workers=dl_cfg['num_workers'],
        pin_memory=dl_cfg['pin_memory'],
    )

    model = DenoisingAutoencoder(in_channels=3, base_filters=32).to(device)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Model parameters: {total_params:,}")

    # Use L1 loss for denoising — more robust to outliers than MSE
    criterion = nn.L1Loss()

    optimizer = optim.Adam(
        model.parameters(),
        lr=train_cfg['learning_rate'],
    )

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=10, verbose=True
    )

    use_amp = train_cfg['use_amp'] and (device.type == 'cuda')
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    checkpoint_dir = os.path.join(paths['checkpoints'], 'denoising')
    val_image_dir = os.path.join(paths['validation_images'], 'denoising')
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(val_image_dir, exist_ok=True)

    start_epoch = 0
    best_psnr = 0.0

    if resume:
        latest = get_latest_checkpoint(checkpoint_dir, prefix='denoising')
        if latest:
            ckpt = load_checkpoint(latest, model, optimizer, scaler, device)
            start_epoch = ckpt['epoch'] + 1
            best_psnr = ckpt['best_psnr']

    logger = TensorBoardLogger(paths['logs'], experiment_name='denoising')

    for epoch in range(start_epoch, train_cfg['epochs']):
        print(f"\n--- Epoch {epoch + 1}/{train_cfg['epochs']} ---")

        model.train()
        tracker = MetricsTracker()
        progress = tqdm(train_loader, desc="  Training", leave=False)

        for batch_idx, (noisy_batch, clean_batch) in enumerate(progress):
            noisy_batch = noisy_batch.to(device, non_blocking=True)
            clean_batch = clean_batch.to(device, non_blocking=True)

            with torch.cuda.amp.autocast(enabled=use_amp):
                pred = model(noisy_batch)
                loss = criterion(pred, clean_batch)

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()

            if train_cfg['clip_grad_norm'] > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg['clip_grad_norm'])

            scaler.step(optimizer)
            scaler.update()

            psnr = calculate_psnr(pred.detach(), clean_batch)
            tracker.update('loss', loss.item(), noisy_batch.size(0))
            tracker.update('psnr', psnr, noisy_batch.size(0))
            progress.set_postfix(loss=f"{loss.item():.4f}", psnr=f"{psnr:.2f}")

            if batch_idx % train_cfg['log_interval'] == 0:
                step = epoch * len(train_loader) + batch_idx
                logger.log_scalar('Train/Loss', loss.item(), step)
                logger.log_scalar('Train/PSNR', psnr, step)

        avg = tracker.summary()
        print(f"  Train — Loss: {avg['loss']:.4f} | PSNR: {avg['psnr']:.2f} dB")

        if (epoch + 1) % train_cfg['val_interval'] == 0:
            val_psnr = validate_denoising(
                model, val_loader, device, use_amp, epoch, val_image_dir, logger
            )
            scheduler.step(val_psnr)

            if val_psnr > best_psnr:
                best_psnr = val_psnr
                save_checkpoint(
                    model, optimizer, epoch, best_psnr, config,
                    os.path.join(checkpoint_dir, 'denoising_best.pth'), scaler
                )
                print(f"  *** New best PSNR: {best_psnr:.2f} dB ***")

        if (epoch + 1) % train_cfg['save_interval'] == 0:
            save_checkpoint(
                model, optimizer, epoch, best_psnr, config,
                os.path.join(checkpoint_dir, f'denoising_epoch_{epoch+1:04d}.pth'), scaler
            )

    logger.close()
    print(f"\n  Training complete. Best PSNR: {best_psnr:.2f} dB")


def validate_denoising(model, val_loader, device, use_amp, epoch, val_image_dir, logger):
    model.eval()
    tracker = MetricsTracker()
    saved_sample = False

    with torch.no_grad():
        for noisy_batch, clean_batch in tqdm(val_loader, desc="  Validation", leave=False):
            noisy_batch = noisy_batch.to(device, non_blocking=True)
            clean_batch = clean_batch.to(device, non_blocking=True)

            with torch.cuda.amp.autocast(enabled=use_amp):
                pred = model(noisy_batch)

            psnr = calculate_psnr(pred, clean_batch)
            ssim = calculate_ssim(pred, clean_batch)
            tracker.update('psnr', psnr, noisy_batch.size(0))
            tracker.update('ssim', ssim, noisy_batch.size(0))

            if not saved_sample:
                for i in range(min(4, pred.size(0))):
                    save_image_tensor(noisy_batch[i], os.path.join(val_image_dir, f"epoch_{epoch+1:04d}_img{i}_noisy.png"))
                    save_image_tensor(pred[i], os.path.join(val_image_dir, f"epoch_{epoch+1:04d}_img{i}_pred.png"))
                    save_image_tensor(clean_batch[i], os.path.join(val_image_dir, f"epoch_{epoch+1:04d}_img{i}_clean.png"))
                saved_sample = True

    avg = tracker.summary()
    print(f"  Val   — PSNR: {avg['psnr']:.2f} dB | SSIM: {avg['ssim']:.4f}")
    logger.log_scalar('Val/PSNR', avg['psnr'], epoch)
    logger.log_scalar('Val/SSIM', avg['ssim'], epoch)
    return avg['psnr']


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/config.yaml')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    train_denoising(args.config, args.resume)