"""
Combined DIV2K + CelebA Dataset.

Merges both datasets into one DataLoader so the model
trains on general content (DIV2K) and faces (CelebA)
simultaneously in every epoch.

This gives the best of both worlds:
  - DIV2K: landscapes, animals, textures, buildings
  - CelebA: faces, skin, hair, eyes

The model sees both types in every training batch.
"""

import os
from torch.utils.data import ConcatDataset
from datasets.div2k_dataset import DIV2KDataset
from datasets.denoising_dataset import DenoisingDataset


def build_combined_sr_dataset(config: dict, split: str = 'train'):
    """
    Build combined SR dataset from DIV2K + CelebA.

    Args:
        config: Loaded config_celeba.yaml dictionary
        split:  'train' or 'val'

    Returns:
        ConcatDataset combining both datasets
    """
    data_cfg  = config['data']
    patch_cfg = config['patches']
    scale     = data_cfg['scale_factor']

    if split == 'train':
        max_div2k  = data_cfg['max_div2k_train']
        max_celeba = data_cfg['max_celeba_train']
        augment    = True
    else:
        max_div2k  = data_cfg['max_div2k_valid']
        max_celeba = data_cfg['max_celeba_valid']
        augment    = False

    # DIV2K portion
    div2k_dataset = DIV2KDataset(
        hr_dir=os.path.join(data_cfg['div2k_root'], data_cfg['div2k_train_hr']
               if split == 'train' else data_cfg['div2k_valid_hr']),
        lr_dir=os.path.join(data_cfg['div2k_root'], data_cfg['div2k_train_lr']
               if split == 'train' else data_cfg['div2k_valid_lr']),
        hr_patch_size=patch_cfg['hr_patch_size'],
        scale=scale,
        max_images=max_div2k,
        augment=augment,
        split=split,
    )

    # CelebA portion
    celeba_dataset = DIV2KDataset(
        hr_dir=os.path.join(data_cfg['celeba_root'], data_cfg['celeba_hr']),
        lr_dir=os.path.join(data_cfg['celeba_root'], data_cfg['celeba_lr']),
        hr_patch_size=patch_cfg['hr_patch_size'],
        scale=scale,
        max_images=max_celeba,
        augment=augment,
        split=split,
    )

    # Combine both
    combined = ConcatDataset([div2k_dataset, celeba_dataset])

    print(f"  [Combined Dataset] {split.upper()}")
    print(f"    DIV2K images  : {len(div2k_dataset)}")
    print(f"    CelebA images : {len(celeba_dataset)}")
    print(f"    Total         : {len(combined)}")

    return combined


def build_combined_denoising_dataset(config: dict, split: str = 'train'):
    """
    Build combined denoising dataset from DIV2K + CelebA HR images.

    For denoising we only need HR images — no LR needed.
    Both DIV2K and CelebA HR images are used as clean targets.
    Noise is added synthetically during training.

    Args:
        config: Loaded config dictionary
        split:  'train' or 'val'

    Returns:
        ConcatDataset combining both datasets
    """
    data_cfg  = config['data']
    patch_cfg = config['patches']

    if split == 'train':
        max_div2k  = data_cfg['max_div2k_train']
        max_celeba = data_cfg['max_celeba_train']
        augment    = True
    else:
        max_div2k  = data_cfg['max_div2k_valid']
        max_celeba = data_cfg['max_celeba_valid']
        augment    = False

    # DIV2K denoising portion
    div2k_denoise = DenoisingDataset(
        hr_dir=os.path.join(data_cfg['div2k_root'],
               data_cfg['div2k_train_hr'] if split == 'train'
               else data_cfg['div2k_valid_hr']),
        patch_size=patch_cfg['hr_patch_size'],
        max_images=max_div2k,
        noise_type='mixed',
        noise_level=0.1,
        jpeg_quality=30,
        augment=augment,
    )

    # CelebA denoising portion
    celeba_denoise = DenoisingDataset(
        hr_dir=os.path.join(data_cfg['celeba_root'], data_cfg['celeba_hr']),
        patch_size=patch_cfg['hr_patch_size'],
        max_images=max_celeba,
        noise_type='mixed',
        noise_level=0.1,
        jpeg_quality=30,
        augment=augment,
    )

    combined = ConcatDataset([div2k_denoise, celeba_denoise])

    print(f"  [Combined Denoising Dataset] {split.upper()}")
    print(f"    DIV2K images  : {len(div2k_denoise)}")
    print(f"    CelebA images : {len(celeba_denoise)}")
    print(f"    Total         : {len(combined)}")

    return combined