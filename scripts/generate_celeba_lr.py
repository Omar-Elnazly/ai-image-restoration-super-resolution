"""
Generates LR (low-resolution) versions of CelebA images
by bicubic downsampling by factor 2.

Creates:
    data/CelebA/img_align_celeba_LR/
        000001.jpg
        000002.jpg
        ...

Run once before training:
    python generate_celeba_lr.py
"""

import os
from PIL import Image
from tqdm import tqdm

# ----------------------------------------------------------------
# Settings
# ----------------------------------------------------------------
HR_DIR    = "data/CelebA/img_align_celeba"
LR_DIR    = "data/CelebA/img_align_celeba_LR"
SCALE     = 2
MAX_IMAGES = None  # Start with 10k for reasonable training time
             # Set to None to use all 202,599 images

# ----------------------------------------------------------------
# Create output directory
# ----------------------------------------------------------------
os.makedirs(LR_DIR, exist_ok=True)

# ----------------------------------------------------------------
# Collect image files
# ----------------------------------------------------------------
all_files = sorted([
    f for f in os.listdir(HR_DIR)
    if f.lower().endswith(('.jpg', '.png'))
])

if MAX_IMAGES is not None:
    all_files = all_files[:MAX_IMAGES]

print(f"\nGenerating LR CelebA images...")
print(f"  HR source : {HR_DIR}")
print(f"  LR output : {LR_DIR}")
print(f"  Scale     : x{SCALE}")
print(f"  Images    : {len(all_files)}")

# ----------------------------------------------------------------
# Generate LR images
# ----------------------------------------------------------------
skipped = 0
for filename in tqdm(all_files):
    hr_path = os.path.join(HR_DIR, filename)
    lr_path = os.path.join(LR_DIR, filename)

    # Skip if already generated
    if os.path.exists(lr_path):
        skipped += 1
        continue

    try:
        hr_img = Image.open(hr_path).convert('RGB')
        w, h = hr_img.size

        # Bicubic downsample
        lr_img = hr_img.resize(
            (w // SCALE, h // SCALE),
            Image.BICUBIC
        )
        lr_img.save(lr_path, quality=95)

    except Exception as e:
        print(f"\n  Warning: Could not process {filename}: {e}")

print(f"\nDone.")
print(f"  Generated : {len(all_files) - skipped}")
print(f"  Skipped   : {skipped} (already existed)")
print(f"  LR folder : {LR_DIR}")

# Verify
lr_files = os.listdir(LR_DIR)
print(f"  Total LR files: {len(lr_files)}")