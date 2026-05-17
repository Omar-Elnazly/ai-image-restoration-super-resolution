"""
SRGAN Training Script — Phase 4

SRGAN is more complex because we have two networks to train alternately:
  1. Generator (SRResNet) — tries to fool the discriminator
  2. Discriminator — tries to distinguish real HR from SR images

GAN training loop:
  Step A: Train discriminator
    - Compute loss on real HR images (label=1) and fake SR images (label=0)
    - Update discriminator weights
  
  Step B: Train generator
    - Generate SR images from LR
    - Compute perceptual loss (VGG + pixel)
    - Compute adversarial loss (discriminator's response to fake images)
    - Update generator weights ONLY

IMPORTANT: We initialize the generator with SRResNet weights pretrained in Phase 3.
           This gives GAN training a better starting point and prevents mode collapse.
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

from datasets.div2k_dataset import DIV2KDataset
from models.srresnet import SRResNet
from models.srgan import SRGANDiscriminator, PerceptualLoss
from utils.metrics import calculate_psnr, calculate_ssim, MetricsTracker
from utils.checkpoint import save_checkpoint, load_checkpoint, get_latest_checkpoint
from utils.logger import TensorBoardLogger
from utils.image_utils import save_image_tensor


def train_srgan(
    config_path: str = "configs/config.yaml",
    resume: bool = False,
    generator_pretrained: str = None,  # Path to SRResNet checkpoint to initialize from
):

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    torch.manual_seed(config['project']['seed'])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\n{'='*50}")
    print(f"  Training SRGAN")
    print(f"  Device: {device}")
    print(f"{'='*50}\n")

    data_cfg = config['data']
    patch_cfg = config['patches']
    dl_cfg = config['dataloader']
    train_cfg = config['training']
    paths = config['paths']
    data_root = data_cfg['root']

    train_dataset = DIV2KDataset(
        hr_dir=os.path.join(data_root, data_cfg['train_hr']),
        lr_dir=os.path.join(data_root, data_cfg['train_lr']),
        hr_patch_size=patch_cfg['hr_patch_size'],
        scale=data_cfg['scale_factor'],
        max_images=data_cfg['max_train_images'],
        augment=True, split='train',
    )

    train_loader = DataLoader(train_dataset, batch_size=dl_cfg['batch_size'],
                              shuffle=True, num_workers=dl_cfg['num_workers'],
                              pin_memory=dl_cfg['pin_memory'], drop_last=True)

    # ---- Models ----
    generator = SRResNet(
        scale_factor=data_cfg['scale_factor'],
        num_channels=3, num_filters=64, num_res_blocks=16,
    ).to(device)

    discriminator = SRGANDiscriminator(in_channels=3).to(device)

    # Load pretrained generator weights (CRITICAL for stable SRGAN training)
    if generator_pretrained and os.path.exists(generator_pretrained):
        ckpt = load_checkpoint(generator_pretrained, generator, device=device)
        print(f"  Generator initialized from: {generator_pretrained}")
    else:
        print("  WARNING: No pretrained generator. Consider training SRResNet first (Phase 3).")

    # ---- Losses ----
    perceptual_loss = PerceptualLoss(pixel_weight=1.0, vgg_weight=0.006).to(device)

    # Binary Cross Entropy with logits — numerically stable GAN loss
    adversarial_criterion = nn.BCEWithLogitsLoss()

    # Adversarial loss weight — keep small so perceptual quality doesn't collapse
    adversarial_weight = 0.001

    # ---- Optimizers ----
    # Separate optimizers for generator and discriminator
    g_optimizer = optim.Adam(generator.parameters(), lr=1e-4, betas=(0.9, 0.999))
    d_optimizer = optim.Adam(discriminator.parameters(), lr=1e-4, betas=(0.9, 0.999))

    use_amp = train_cfg['use_amp'] and (device.type == 'cuda')
    g_scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    d_scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    checkpoint_dir = os.path.join(paths['checkpoints'], 'srgan')
    val_image_dir = os.path.join(paths['validation_images'], 'srgan')
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(val_image_dir, exist_ok=True)

    logger = TensorBoardLogger(paths['logs'], experiment_name='srgan')

    best_psnr = 0.0
    start_epoch = 0

    for epoch in range(start_epoch, train_cfg['epochs']):
        print(f"\n--- Epoch {epoch + 1}/{train_cfg['epochs']} ---")

        generator.train()
        discriminator.train()
        tracker = MetricsTracker()
        progress = tqdm(train_loader, desc="  SRGAN Training", leave=False)

        for batch_idx, (lr_batch, hr_batch) in enumerate(progress):
            lr_batch = lr_batch.to(device, non_blocking=True)
            hr_batch = hr_batch.to(device, non_blocking=True)
            batch_size = lr_batch.size(0)

            # ==============================================================
            # Step A: Train Discriminator
            # ==============================================================
            d_optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=use_amp):
                # Generate SR images (detached — we don't want generator gradients here)
                sr_images = generator(lr_batch).detach()

                # Real images should be classified as 1
                # Use soft labels (0.9 instead of 1.0) for training stability
                real_labels = torch.ones(batch_size, 1, device=device) * 0.9
                fake_labels = torch.zeros(batch_size, 1, device=device)

                d_real = discriminator(hr_batch)
                d_fake = discriminator(sr_images)

                d_loss_real = adversarial_criterion(d_real, real_labels)
                d_loss_fake = adversarial_criterion(d_fake, fake_labels)
                d_loss = (d_loss_real + d_loss_fake) / 2

            d_scaler.scale(d_loss).backward()
            d_scaler.step(d_optimizer)
            d_scaler.update()

            # ==============================================================
            # Step B: Train Generator
            # ==============================================================
            g_optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=use_amp):
                sr_images = generator(lr_batch)

                # Perceptual loss (content loss)
                content_loss = perceptual_loss(sr_images, hr_batch)

                # Adversarial loss: generator wants discriminator to output 1 for SR
                g_adversarial = discriminator(sr_images)
                g_adv_loss = adversarial_criterion(
                    g_adversarial,
                    torch.ones(batch_size, 1, device=device)  # Generator wants "real" labels
                )

                g_loss = content_loss + adversarial_weight * g_adv_loss

            g_scaler.scale(g_loss).backward()

            if train_cfg['clip_grad_norm'] > 0:
                g_scaler.unscale_(g_optimizer)
                torch.nn.utils.clip_grad_norm_(generator.parameters(), train_cfg['clip_grad_norm'])

            g_scaler.step(g_optimizer)
            g_scaler.update()

            # Metrics
            psnr = calculate_psnr(sr_images.detach(), hr_batch)
            tracker.update('g_loss', g_loss.item(), batch_size)
            tracker.update('d_loss', d_loss.item(), batch_size)
            tracker.update('psnr', psnr, batch_size)
            progress.set_postfix(g_loss=f"{g_loss.item():.4f}", d_loss=f"{d_loss.item():.4f}", psnr=f"{psnr:.2f}")

            if batch_idx % train_cfg['log_interval'] == 0:
                step = epoch * len(train_loader) + batch_idx
                logger.log_scalar('Train/G_Loss', g_loss.item(), step)
                logger.log_scalar('Train/D_Loss', d_loss.item(), step)
                logger.log_scalar('Train/PSNR', psnr, step)

        avg = tracker.summary()
        print(f"  Train — G_Loss: {avg['g_loss']:.4f} | D_Loss: {avg['d_loss']:.4f} | PSNR: {avg['psnr']:.2f} dB")

        # Save sample validation images every 5 epochs
        if (epoch + 1) % 5 == 0:
            generator.eval()
            with torch.no_grad():
                lr_sample, hr_sample = next(iter(train_loader))
                lr_sample = lr_sample[:4].to(device)
                hr_sample = hr_sample[:4].to(device)
                with torch.cuda.amp.autocast(enabled=use_amp):
                    sr_sample = generator(lr_sample)
                for i in range(min(4, sr_sample.size(0))):
                    save_image_tensor(lr_sample[i], os.path.join(val_image_dir, f"epoch_{epoch+1:04d}_img{i}_lr.png"))
                    save_image_tensor(sr_sample[i], os.path.join(val_image_dir, f"epoch_{epoch+1:04d}_img{i}_sr.png"))
                    save_image_tensor(hr_sample[i], os.path.join(val_image_dir, f"epoch_{epoch+1:04d}_img{i}_hr.png"))

        # Save checkpoints
        if (epoch + 1) % train_cfg['save_interval'] == 0:
            save_checkpoint(generator, g_optimizer, epoch, best_psnr, config,
                            os.path.join(checkpoint_dir, f'srgan_gen_epoch_{epoch+1:04d}.pth'), g_scaler)
            save_checkpoint(discriminator, d_optimizer, epoch, best_psnr, config,
                            os.path.join(checkpoint_dir, f'srgan_disc_epoch_{epoch+1:04d}.pth'), d_scaler)

    logger.close()
    print("\n  SRGAN Training complete.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/config.yaml')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--generator_pretrained', type=str, default=None,
                        help='Path to SRResNet checkpoint to initialize generator')
    args = parser.parse_args()
    train_srgan(args.config, args.resume, args.generator_pretrained)