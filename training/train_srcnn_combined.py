"""
SRCNN Combined Training — DIV2K + CelebA

Loads pretrained DIV2K SRCNN weights and fine-tunes
on combined DIV2K + CelebA dataset.

Original checkpoints are never touched.

Run:
    python training/train_srcnn_combined.py
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

from datasets.combined_dataset import build_combined_sr_dataset
from models.srcnn import build_srcnn
from utils.metrics import calculate_psnr, calculate_ssim, MetricsTracker
from utils.checkpoint import save_checkpoint, load_checkpoint
from utils.logger import TensorBoardLogger
from utils.image_utils import save_image_tensor


def train_srcnn_combined(
    config_path: str = "configs/config_celeba.yaml",
    pretrained_path: str = "checkpoints/srcnn/srcnn_best.pth",
):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    torch.manual_seed(config['project']['seed'])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\n{'='*50}")
    print(f"  SRCNN Combined Training (DIV2K + CelebA)")
    print(f"  Device: {device}")
    print(f"  Pretrained: {pretrained_path}")
    print(f"{'='*50}\n")

    dl_cfg    = config['dataloader']
    train_cfg = config['training']
    paths     = config['paths']
    scale     = config['data']['scale_factor']

    # Build combined datasets
    train_dataset = build_combined_sr_dataset(config, split='train')
    val_dataset   = build_combined_sr_dataset(config, split='val')

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

    # Build model and load pretrained DIV2K weights
    model = build_srcnn(config).to(device)

    if os.path.exists(pretrained_path):
        load_checkpoint(pretrained_path, model, device=device)
        print(f"\n  Pretrained weights loaded successfully.")
        print(f"  Fine-tuning on DIV2K + CelebA combined.\n")
    else:
        print(f"  No pretrained found — training from scratch.\n")

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=train_cfg['learning_rate'])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5, verbose=True
    )

    use_amp = train_cfg['use_amp'] and (device.type == 'cuda')
    scaler  = torch.cuda.amp.GradScaler(enabled=use_amp)

    checkpoint_dir = os.path.join(paths['checkpoints'], 'srcnn')
    val_image_dir  = os.path.join(paths['validation_images'], 'srcnn')
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(val_image_dir, exist_ok=True)

    best_psnr = 0.0
    logger    = TensorBoardLogger(paths['logs'], 'srcnn_combined')

    for epoch in range(train_cfg['epochs']):
        print(f"\n--- Epoch {epoch+1}/{train_cfg['epochs']} ---")

        # Training
        model.train()
        tracker  = MetricsTracker()
        progress = tqdm(train_loader, desc="  Training", leave=False)

        for batch_idx, (lr_batch, hr_batch) in enumerate(progress):
            lr_batch = lr_batch.to(device, non_blocking=True)
            hr_batch = hr_batch.to(device, non_blocking=True)

            lr_upscaled = torch.nn.functional.interpolate(
                lr_batch, scale_factor=scale,
                mode='bicubic', align_corners=False
            )

            with torch.cuda.amp.autocast(enabled=use_amp):
                pred = model(lr_upscaled)
                loss = criterion(pred, hr_batch)

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            psnr = calculate_psnr(pred.detach(), hr_batch)
            tracker.update('loss', loss.item(), lr_batch.size(0))
            tracker.update('psnr', psnr, lr_batch.size(0))
            progress.set_postfix(loss=f"{loss.item():.4f}", psnr=f"{psnr:.2f}")

            if batch_idx % train_cfg['log_interval'] == 0:
                step = epoch * len(train_loader) + batch_idx
                logger.log_scalar('Train/Loss', loss.item(), step)
                logger.log_scalar('Train/PSNR', psnr, step)

        avg = tracker.summary()
        print(f"  Train — Loss: {avg['loss']:.4f} | PSNR: {avg['psnr']:.2f} dB")

        # Validation
        model.eval()
        val_tracker  = MetricsTracker()
        saved_sample = False

        with torch.no_grad():
            for lr_batch, hr_batch in tqdm(val_loader, desc="  Validation", leave=False):
                lr_batch = lr_batch.to(device, non_blocking=True)
                hr_batch = hr_batch.to(device, non_blocking=True)

                lr_upscaled = torch.nn.functional.interpolate(
                    lr_batch, scale_factor=scale,
                    mode='bicubic', align_corners=False
                )

                with torch.cuda.amp.autocast(enabled=use_amp):
                    pred = model(lr_upscaled)

                val_tracker.update('psnr', calculate_psnr(pred, hr_batch), lr_batch.size(0))
                val_tracker.update('ssim', calculate_ssim(pred, hr_batch), lr_batch.size(0))

                if not saved_sample:
                    for i in range(min(4, pred.size(0))):
                        save_image_tensor(
                            pred[i],
                            os.path.join(val_image_dir, f"epoch_{epoch+1:04d}_img{i}_pred.png")
                        )
                        save_image_tensor(
                            hr_batch[i],
                            os.path.join(val_image_dir, f"epoch_{epoch+1:04d}_img{i}_hr.png")
                        )
                    saved_sample = True

        val_avg = val_tracker.summary()
        print(f"  Val   — PSNR: {val_avg['psnr']:.2f} dB | SSIM: {val_avg['ssim']:.4f}")

        logger.log_scalar('Val/PSNR', val_avg['psnr'], epoch)
        logger.log_scalar('Val/SSIM', val_avg['ssim'], epoch)
        scheduler.step(val_avg['psnr'])

        if val_avg['psnr'] > best_psnr:
            best_psnr = val_avg['psnr']
            save_checkpoint(
                model, optimizer, epoch, best_psnr, config,
                os.path.join(checkpoint_dir, 'srcnn_combined_best.pth'), scaler
            )
            print(f"  *** New best PSNR: {best_psnr:.2f} dB ***")

        if (epoch + 1) % train_cfg['save_interval'] == 0:
            save_checkpoint(
                model, optimizer, epoch, best_psnr, config,
                os.path.join(checkpoint_dir, f'srcnn_combined_epoch_{epoch+1:04d}.pth'),
                scaler
            )

    logger.close()
    print(f"\n  Combined SRCNN training complete.")
    print(f"  Best PSNR : {best_psnr:.2f} dB")
    print(f"  Saved to  : {checkpoint_dir}/srcnn_combined_best.pth")
    print(f"  Original  : checkpoints/srcnn/srcnn_best.pth (untouched)")


if __name__ == "__main__":
    train_srcnn_combined()