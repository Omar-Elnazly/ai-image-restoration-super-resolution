"""
Test Image Generator for All 4 Models

Generates all test images needed to demonstrate every model
using a single source image provided by the user.

Generates:
  For Denoising Autoencoder:
    - clean_original.png         (clean reference)
    - noisy_mild.png             (Gaussian noise σ=20)
    - noisy_heavy.png            (Gaussian noise σ=50)
    - jpeg_artifacts.png         (16x16 block artifacts — guaranteed visible)

  For Super-Resolution (SRCNN, SRResNet, SRGAN):
    - sr_input_lr.png            (downscaled 50% — model input)
    - sr_reference_hr.png        (original full size — reference)

  For Chain Pipeline (Denoise → SR):
    - chain_input_noisy_lr.png   (downscaled + noisy — worst case input)

Run from project root:
    python scripts/make_test_noise.py --image test_images/original.jpg

Or use default path:
    python scripts/make_test_noise.py
"""

import os
import sys
import argparse
import numpy as np
from PIL import Image
import io

# ----------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------
OUTPUT_DIR  = "test_images"
TARGET_SIZE = 512
LR_SCALE    = 2


def load_and_prepare(image_path: str, size: int) -> Image.Image:
    """
    Load image, convert to RGB, and resize to square target size.
    Keeps the most centered crop to avoid black borders.
    """
    img = Image.open(image_path).convert("RGB")
    w, h = img.size

    # Center crop to square
    min_dim = min(w, h)
    left    = (w - min_dim) // 2
    top     = (h - min_dim) // 2
    img     = img.crop((left, top, left + min_dim, top + min_dim))

    # Resize to target size
    img = img.resize((size, size), Image.LANCZOS)
    return img


def add_block_artifacts(img: Image.Image, block_size: int = 16) -> Image.Image:
    """
    Create visible block artifacts by averaging pixels within blocks.

    This is more reliable than JPEG compression for demonstration
    because the blocks are guaranteed to be visible regardless of
    image content. Rich textures like fur can hide JPEG blocks,
    but block averaging always produces clear square patches.

    How it works:
      Every block_size x block_size region is replaced with its
      mean color — creating flat colored squares across the image.
      This simulates heavy compression artifacts in a controlled way.

    Args:
        img:        Input PIL image
        block_size: Size of each square block in pixels
                    16 = very visible, 8 = subtle

    Returns:
        PIL Image with clear square block artifacts
    """
    arr    = np.array(img).astype(np.float32)
    result = arr.copy()
    h, w   = arr.shape[:2]

    for y in range(0, h, block_size):
        for x in range(0, w, block_size):
            y_end = min(y + block_size, h)
            x_end = min(x + block_size, w)

            # Replace entire block with its mean color
            block_mean               = arr[y:y_end, x:x_end].mean(axis=(0, 1))
            result[y:y_end, x:x_end] = block_mean

    return Image.fromarray(result.astype(np.uint8))


def generate_denoising_images(img: Image.Image, output_dir: str):
    """
    Generate all test images for the Denoising Autoencoder.

    Creates 4 images:
      clean_original  — untouched source (ground truth)
      noisy_mild      — light Gaussian noise σ=20 (easy case)
      noisy_heavy     — heavy Gaussian noise σ=50 (hard case)
      jpeg_artifacts  — 16x16 block averaging (guaranteed visible artifacts)

    Why block averaging instead of JPEG compression:
      Images with rich textures (fur, hair, feathers) naturally hide
      JPEG blocking artifacts. Block averaging guarantees visible
      square patches regardless of image content while preserving
      the original colors correctly.
    """
    print("\n--- Generating Denoising Test Images ---")

    arr = np.array(img).astype(np.float32)

    # ---- Clean original ----
    clean_path = os.path.join(output_dir, "clean_original.png")
    img.save(clean_path)
    print(f"  Saved: {clean_path}")

    # ---- Mild Gaussian noise (σ=20) ----
    # Simulates low-light camera sensor noise
    # Fixed seed for reproducibility
    np.random.seed(42)
    noise_mild = np.random.normal(0, 20, arr.shape)
    noisy_mild = np.clip(arr + noise_mild, 0, 255).astype(np.uint8)
    mild_path  = os.path.join(output_dir, "noisy_mild.png")
    Image.fromarray(noisy_mild).save(mild_path)
    print(f"  Saved: {mild_path}")

    # ---- Heavy Gaussian noise (σ=50) ----
    # Simulates very dark environment or heavily damaged photo
    noise_heavy = np.random.normal(0, 50, arr.shape)
    noisy_heavy = np.clip(arr + noise_heavy, 0, 255).astype(np.uint8)
    heavy_path  = os.path.join(output_dir, "noisy_heavy.png")
    Image.fromarray(noisy_heavy).save(heavy_path)
    print(f"  Saved: {heavy_path}")

    # ---- Block artifacts (simulated compression) ----
    # Uses 16x16 block averaging — visible on any image type
    blocked_img  = add_block_artifacts(img, block_size=16)
    blocked_path = os.path.join(output_dir, "jpeg_artifacts.png")
    blocked_img.save(blocked_path)
    print(f"  Saved: {blocked_path}")
    print(f"  Note: 16x16 block averaging — artifacts clearly visible")


def generate_sr_images(img: Image.Image, output_dir: str, scale: int):
    """
    Generate matched LR/HR pair for Super-Resolution testing.

    LR is created by bicubic downscaling the HR image so both are
    perfectly aligned. This is the correct way to create test pairs
    and matches how DIV2K dataset LR images were generated.

    Creates 2 images:
      sr_reference_hr  — full size clean image (ground truth)
      sr_input_lr      — downscaled by scale factor (model input)
    """
    print("\n--- Generating Super-Resolution Test Images ---")

    # HR reference
    hr_path = os.path.join(output_dir, "sr_reference_hr.png")
    img.save(hr_path)
    print(f"  Saved: {hr_path}  (size: {img.size[0]}x{img.size[1]})")

    # LR input — bicubic downsample
    lr_size = (img.size[0] // scale, img.size[1] // scale)
    lr_img  = img.resize(lr_size, Image.BICUBIC)
    lr_path = os.path.join(output_dir, "sr_input_lr.png")
    lr_img.save(lr_path)
    print(f"  Saved: {lr_path}  (size: {lr_img.size[0]}x{lr_img.size[1]})")

    # Verify pair is correctly matched
    hr_check    = Image.open(hr_path)
    lr_check    = Image.open(lr_path)
    expected_lr = (hr_check.size[0] // scale, hr_check.size[1] // scale)

    if lr_check.size == expected_lr:
        print(f"  Pair verified: HR {hr_check.size} → LR {lr_check.size} ✓")
    else:
        print(f"  WARNING: Size mismatch — HR {hr_check.size}, LR {lr_check.size}")

    print(f"\n  How to use:")
    print(f"    Upload sr_input_lr.png     → Input Image")
    print(f"    Upload sr_reference_hr.png → Reference Image (optional)")
    print(f"    Select SRCNN / SRResNet / SRGAN")
    print(f"    Click Run Enhancement")


def generate_chain_pipeline_image(img: Image.Image, output_dir: str, scale: int):
    """
    Generate test image for the Chain Pipeline demo (Denoise → SR).

    Creates 1 image:
      chain_input_noisy_lr — downscaled AND noisy
                             represents worst case: both degradations at once

    Why this matters:
      Running denoising BEFORE super-resolution produces better results
      than running SR on a noisy image. This test demonstrates that
      combining specialized models outperforms a single model.

    Demo workflow:
      Step 1: Upload chain_input_noisy_lr → Denoising → Save output
      Step 2: Upload saved output → SRResNet or SRGAN → Save final
      Step 3: Compare final output vs sr_reference_hr
    """
    print("\n--- Generating Chain Pipeline Test Image ---")

    arr = np.array(img).astype(np.float32)

    # Add moderate noise to full size image
    np.random.seed(99)
    noise     = np.random.normal(0, 25, arr.shape)
    noisy_arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
    noisy_img = Image.fromarray(noisy_arr)

    # Downscale the noisy image
    lr_size      = (noisy_img.size[0] // scale, noisy_img.size[1] // scale)
    noisy_lr_img = noisy_img.resize(lr_size, Image.BICUBIC)
    chain_path   = os.path.join(output_dir, "chain_input_noisy_lr.png")
    noisy_lr_img.save(chain_path)

    print(f"  Saved: {chain_path}  (size: {noisy_lr_img.size[0]}x{noisy_lr_img.size[1]})")
    print(f"\n  How to use:")
    print(f"    Step 1: chain_input_noisy_lr.png → Denoising Autoencoder → Save output")
    print(f"    Step 2: Saved output → SRResNet or SRGAN → Save final")
    print(f"    Step 3: Compare final vs sr_reference_hr.png")


def print_usage_guide(output_dir: str):
    """Print a complete guide for using each generated test image."""
    print(f"\n{'='*60}")
    print(f"  ALL TEST IMAGES GENERATED SUCCESSFULLY")
    print(f"  Output folder: {output_dir}/")
    print(f"{'='*60}")
    print(f"""
  HOW TO TEST EACH MODEL IN THE UI (http://localhost:7860)
  ─────────────────────────────────────────────────────────

  MODEL 1 — SRCNN
    Input     : sr_input_lr.png        (256x256)
    Reference : sr_reference_hr.png    (512x512, optional)
    Model     : SRCNN (Super-Resolution x2)
    Expect    : Output 512x512, sharper edges than input

  MODEL 2 — DENOISING AUTOENCODER
    Test A — Mild noise:
      Input     : noisy_mild.png
      Reference : clean_original.png
      Expect    : Light grain removed, colors and structure preserved

    Test B — Heavy noise:
      Input     : noisy_heavy.png
      Reference : clean_original.png
      Expect    : Strong colored noise removed, subject still clear

    Test C — Block artifacts:
      Input     : jpeg_artifacts.png
      Reference : clean_original.png
      Expect    : Square blocks smoothed out, edges restored

  MODEL 3 — SRResNet
    Input     : sr_input_lr.png        (256x256)
    Reference : sr_reference_hr.png    (512x512, optional)
    Model     : SRResNet (Super-Resolution x2)
    Expect    : Sharper fur detail than SRCNN

  MODEL 4 — SRGAN
    Input     : sr_input_lr.png        (256x256)
    Reference : sr_reference_hr.png    (512x512, optional)
    Model     : SRGAN (Super-Resolution x2)
    Expect    : Most realistic looking output, sharpest fur texture

  CHAIN PIPELINE — Best Demo for Presentation
    Step 1 : chain_input_noisy_lr.png → Denoising → Save output
    Step 2 : Saved output → SRResNet → Save final output
    Step 3 : Compare final vs sr_reference_hr.png

  ─────────────────────────────────────────────────────────
  Generated files summary:
    clean_original.png         Clean ground truth (512x512)
    noisy_mild.png             Light noise σ=20   (512x512)
    noisy_heavy.png            Heavy noise σ=50   (512x512)
    jpeg_artifacts.png         Block artifacts    (512x512)
    sr_reference_hr.png        SR ground truth    (512x512)
    sr_input_lr.png            SR model input     (256x256)
    chain_input_noisy_lr.png   Chain demo input   (256x256)
""")


def main():
    parser = argparse.ArgumentParser(
        description="Generate test images for all 4 models"
    )
    parser.add_argument(
        '--image',
        type=str,
        default='test_images/original.jpg',
        help='Path to source image (default: test_images/original.jpg)'
    )
    parser.add_argument(
        '--size',
        type=int,
        default=512,
        help='Output image size in pixels (default: 512)'
    )
    parser.add_argument(
        '--scale',
        type=int,
        default=2,
        help='SR downscale factor (default: 2)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='test_images',
        help='Output directory (default: test_images)'
    )
    args = parser.parse_args()

    # Verify source image exists
    if not os.path.exists(args.image):
        print(f"\n  ERROR: Image not found: {args.image}")
        print(f"  Place your image at: {args.image}")
        print(f"  Or specify a path: --image path/to/image.jpg")
        sys.exit(1)

    # Create output directory
    os.makedirs(args.output, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  TEST IMAGE GENERATOR")
    print(f"{'='*60}")
    print(f"  Source image : {args.image}")
    print(f"  Output size  : {args.size}x{args.size} px")
    print(f"  SR scale     : x{args.scale}")
    print(f"  Output dir   : {args.output}/")

    # Load and prepare source image
    print(f"\n  Loading image...")
    img = load_and_prepare(args.image, args.size)
    print(f"  Image loaded and prepared: {img.size[0]}x{img.size[1]} px RGB")

    # Generate all test images
    generate_denoising_images(img, args.output)
    generate_sr_images(img, args.output, args.scale)
    generate_chain_pipeline_image(img, args.output, args.scale)

    # Print usage guide
    print_usage_guide(args.output)


if __name__ == "__main__":
    main()