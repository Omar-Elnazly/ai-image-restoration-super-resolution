"""
SRGAN Combined Training — DIV2K + CelebA

Loads pretrained combined SRResNet weights as generator
and fine-tunes SRGAN on combined DIV2K + CelebA dataset.

IMPORTANT: Run train_srresnet_combined.py first.
           SRGAN must initialize from a trained generator.

Original checkpoints are never touched.

Run:
    python training/train_srgan_combined.py
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
from models.srresnet import SRResNet
from models.srgan import SRGANDiscriminator, PerceptualLoss
from utils.metrics import calculate_psnr, MetricsTracker
from utils.checkpoint import save_checkpoint, load_checkpoint
from utils.logger import TensorBoardLogger
from utils.image_utils import save_image_tensor


def train_srgan_combined(
    config_path: str = "configs/config_celeba.yaml",

    # Use the combined SRResNet as starting point
    # Falls back to DIV2K SRResNet if combined not ready yet
    generator_pretrained: str = "checkpoints_combined/srresnet/srresnet_combined_best.pth",
    generator_fallback: str = "checkpoints/srresnet/srresnet_best.pth",
):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    torch.manual_seed(config['project']['seed'])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\n{'='*50}")
    print(f"  SRGAN Combined Training (DIV2K + CelebA)")
    print(f"  Device: {device}")
    print(f"{'='*50}\n")

    dl_cfg    = config['dataloader']
    train_cfg = config['training']
    paths     = config['paths']
    scale     = config['data']['scale_factor']

    # Build combined dataset — training only for SRGAN
    train_dataset = build_combined_sr_dataset(config, split='train')

    train_loader = DataLoader(
        train_dataset,
        batch_size=dl_cfg['batch_size'],
        shuffle=True,
        num_workers=dl_cfg['num_workers'],
        pin_memory=dl_cfg['pin_memory'],
        drop_last=True,
    )

    # Build generator and discriminator
    generator = SRResNet(
        scale_factor=scale,
        num_channels=3,
        num_filters=64,
        num_res_blocks=16,
    ).to(device)

    discriminator = SRGANDiscriminator(in_channels=3).to(device)

    # Load generator weights
    # Prefer combined SRResNet, fall back to DIV2K SRResNet
    if os.path.exists(generator_pretrained):
        load_checkpoint(generator_pretrained, generator, device=device)
        print(f"  Generator initialized from combined SRResNet: {generator_pretrained}")
    elif os.path.exists(generator_fallback):
        load_checkpoint(generator_fallback, generator, device=device)
        print(f"  Generator initialized from DIV2K SRResNet: {generator_fallback}")
    else:
        print("  WARNING: No pretrained generator found.")
        print("  Run train_srresnet_combined.py first for best results.")

    # Losses
    perceptual_loss      = PerceptualLoss(pixel_weight=1.0, vgg_weight=0.006).to(device)
    adversarial_criterion = nn.BCEWithLogitsLoss()
    adversarial_weight   = 0.001

    # Separate optimizers for generator and discriminator
    g_optimizer = optim.Adam(generator.parameters(),     lr=train_cfg['learning_rate'], betas=(0.9, 0.999))
    d_optimizer = optim.Adam(discriminator.parameters(), lr=train_cfg['learning_rate'], betas=(0.9, 0.999))

    use_amp  = train_cfg['use_amp'] and (device.type == 'cuda')
    g_scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    d_scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    checkpoint_dir = os.path.join(paths['checkpoints'], 'srgan')
    val_image_dir  = os.path.join(paths['validation_images'], 'srgan')
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(val_image_dir, exist_ok=True)

    best_psnr = 0.0
    logger    = TensorBoardLogger(paths['logs'], 'srgan_combined')

    for epoch in range(train_cfg['epochs']):
        print(f"\n--- Epoch {epoch+1}/{train_cfg['epochs']} ---")

        generator.train()
        discriminator.train()
        tracker  = MetricsTracker()
        progress = tqdm(train_loader, desc="  SRGAN Training", leave=False)

        for batch_idx, (lr_batch, hr_batch) in enumerate(progress):
            lr_batch   = lr_batch.to(device, non_blocking=True)
            hr_batch   = hr_batch.to(device, non_blocking=True)
            batch_size = lr_batch.size(0)

            # ------------------------------------------------
            # Step A: Train Discriminator
            # ------------------------------------------------
            d_optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=use_amp):
                sr_images   = generator(lr_batch).detach()
                real_labels = torch.ones(batch_size, 1, device=device) * 0.9
                fake_labels = torch.zeros(batch_size, 1, device=device)

                d_real      = discriminator(hr_batch)
                d_fake      = discriminator(sr_images)
                d_loss_real = adversarial_criterion(d_real, real_labels)
                d_loss_fake = adversarial_criterion(d_fake, fake_labels)
                d_loss      = (d_loss_real + d_loss_fake) / 2

            d_scaler.scale(d_loss).backward()
            d_scaler.step(d_optimizer)
            d_scaler.update()

            # ------------------------------------------------
            # Step B: Train Generator
            # ------------------------------------------------
            g_optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=use_amp):
                sr_images    = generator(lr_batch)
                content_loss = perceptual_loss(sr_images, hr_batch)
                g_adversarial = discriminator(sr_images)
                g_adv_loss   = adversarial_criterion(
                    g_adversarial,
                    torch.ones(batch_size, 1, device=device)
                )
                g_loss = content_loss + adversarial_weight * g_adv_loss

            g_scaler.scale(g_loss).backward()

            if train_cfg['clip_grad_norm'] > 0:
                g_scaler.unscale_(g_optimizer)
                torch.nn.utils.clip_grad_norm_(
                    generator.parameters(), train_cfg['clip_grad_norm']
                )

            g_scaler.step(g_optimizer)
            g_scaler.update()

            # Metrics
            psnr = calculate_psnr(sr_images.detach(), hr_batch)
            tracker.update('g_loss', g_loss.item(), batch_size)
            tracker.update('d_loss', d_loss.item(), batch_size)
            tracker.update('psnr',   psnr,           batch_size)

            progress.set_postfix(
                g_loss=f"{g_loss.item():.4f}",
                d_loss=f"{d_loss.item():.4f}",
                psnr=f"{psnr:.2f}"
            )

            if batch_idx % train_cfg['log_interval'] == 0:
                step = epoch * len(train_loader) + batch_idx
                logger.log_scalar('Train/G_Loss', g_loss.item(), step)
                logger.log_scalar('Train/D_Loss', d_loss.item(), step)
                logger.log_scalar('Train/PSNR',   psnr,          step)

        avg = tracker.summary()
        print(f"  Train — G_Loss: {avg['g_loss']:.4f} | D_Loss: {avg['d_loss']:.4f} | PSNR: {avg['psnr']:.2f} dB")

        # Track best PSNR and save best generator
        if avg['psnr'] > best_psnr:
            best_psnr = avg['psnr']
            save_checkpoint(
                generator, g_optimizer, epoch, best_psnr, config,
                os.path.join(checkpoint_dir, 'srgan_combined_gen_best.pth'), g_scaler
            )
            print(f"  *** New best PSNR: {best_psnr:.2f} dB ***")

        # Save periodic checkpoints every 5 epochs
        if (epoch + 1) % train_cfg['save_interval'] == 0:
            save_checkpoint(
                generator, g_optimizer, epoch, best_psnr, config,
                os.path.join(checkpoint_dir, f'srgan_combined_gen_epoch_{epoch+1:04d}.pth'),
                g_scaler
            )
            save_checkpoint(
                discriminator, d_optimizer, epoch, best_psnr, config,
                os.path.join(checkpoint_dir, f'srgan_combined_disc_epoch_{epoch+1:04d}.pth'),
                d_scaler
            )

            # Save sample images every 5 epochs
            generator.eval()
            with torch.no_grad():
                lr_sample, hr_sample = next(iter(train_loader))
                lr_sample = lr_sample[:4].to(device)
                hr_sample = hr_sample[:4].to(device)
                with torch.cuda.amp.autocast(enabled=use_amp):
                    sr_sample = generator(lr_sample)
                for i in range(min(4, sr_sample.size(0))):
                    save_image_tensor(
                        lr_sample[i],
                        os.path.join(val_image_dir, f"epoch_{epoch+1:04d}_img{i}_lr.png")
                    )
                    save_image_tensor(
                        sr_sample[i],
                        os.path.join(val_image_dir, f"epoch_{epoch+1:04d}_img{i}_sr.png")
                    )
                    save_image_tensor(
                        hr_sample[i],
                        os.path.join(val_image_dir, f"epoch_{epoch+1:04d}_img{i}_hr.png")
                    )
            generator.train()

    logger.close()
    print(f"\n  Combined SRGAN training complete.")
    print(f"  Best PSNR : {best_psnr:.2f} dB")
    print(f"  Saved to  : {checkpoint_dir}/srgan_combined_gen_best.pth")
    print(f"  Original  : checkpoints/srgan/srgan_gen_epoch_0090.pth (untouched)")


if __name__ == "__main__":
    train_srgan_combined()