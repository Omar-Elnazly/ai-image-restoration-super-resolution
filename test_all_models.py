"""
Comprehensive visual test for all 4 trained models.

This script:
  1. Takes a real test image
  2. Runs all 4 models on it
  3. Saves side-by-side comparison images
  4. Prints a final metrics table

Run from project root:
    python test_all_models.py --image path/to/your/test_image.jpg
    
Or test on a DIV2K validation image:
    python test_all_models.py --use_div2k
"""

import os
import sys
import argparse
import torch
import numpy as np
from PIL import Image
import torchvision.transforms.functional as TF
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend — works without display
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from inference.inference_pipeline import InferencePipeline
from utils.metrics import calculate_psnr, calculate_ssim


# ----------------------------------------------------------------
# Checkpoint paths — update if your filenames differ
# ----------------------------------------------------------------
CHECKPOINTS = {
    'srcnn':     'checkpoints/srcnn/srcnn_best.pth',
    'denoising': 'checkpoints/denoising/denoising_best.pth',
    'srresnet':  'checkpoints/srresnet/srresnet_best.pth',
    'srgan':     'checkpoints/srgan/srgan_gen_epoch_0100.pth',
}

OUTPUT_DIR = 'outputs/model_comparison'


def load_all_pipelines() -> dict:
    """Load all 4 model pipelines. Skip any missing checkpoints."""
    pipelines = {}
    for name, ckpt_path in CHECKPOINTS.items():
        if not os.path.exists(ckpt_path):
            print(f"  [SKIP] {name} — checkpoint not found: {ckpt_path}")
            continue
        try:
            pipelines[name] = InferencePipeline(
                model_type=name,
                checkpoint_path=ckpt_path,
                config_path='configs/config.yaml',
                device='auto',
            )
            print(f"  [OK]   {name} loaded")
        except Exception as e:
            print(f"  [FAIL] {name} — {e}")
    return pipelines


def test_super_resolution(pipelines: dict, lr_image_path: str, hr_image_path: str = None):
    """
    Test SRCNN, SRResNet, SRGAN on a low-resolution image.
    Compares all three outputs side by side.
    """
    print(f"\n--- Super-Resolution Test ---")
    print(f"  Input: {lr_image_path}")

    lr_image = Image.open(lr_image_path).convert('RGB')
    hr_image = Image.open(hr_image_path).convert('RGB') if hr_image_path else None

    sr_models = ['srcnn', 'srresnet', 'srgan']
    results = {}
    metrics = {}

    for model_name in sr_models:
        if model_name not in pipelines:
            continue
        print(f"  Running {model_name}...")
        result = pipelines[model_name].run(
            input_image=lr_image,
            target_image=hr_image,
        )
        results[model_name] = result['output']
        if 'psnr' in result:
            metrics[model_name] = {
                'psnr': result['psnr'],
                'ssim': result['ssim'],
            }

    # Build comparison figure
    num_cols = 2 + len(results)  # LR + HR (optional) + model outputs
    fig, axes = plt.subplots(1, num_cols, figsize=(4 * num_cols, 5))
    fig.suptitle('Super-Resolution Comparison', fontsize=14, fontweight='bold')

    col = 0

    # Show LR input (bicubic upscaled for fair visual comparison)
    lr_upscaled = lr_image.resize(
        (lr_image.width * 2, lr_image.height * 2), Image.BICUBIC
    )
    axes[col].imshow(lr_upscaled)
    axes[col].set_title('Bicubic Baseline\n(Input x2)', fontsize=10)
    axes[col].axis('off')
    col += 1

    # Show each model output
    for model_name, output_img in results.items():
        axes[col].imshow(output_img)
        title = model_name.upper()
        if model_name in metrics:
            title += f"\nPSNR: {metrics[model_name]['psnr']:.2f} dB"
            title += f"\nSSIM: {metrics[model_name]['ssim']:.4f}"
        axes[col].set_title(title, fontsize=10)
        axes[col].axis('off')
        col += 1

    # Show HR ground truth if provided
    if hr_image:
        axes[col].imshow(hr_image)
        axes[col].set_title('Ground Truth\n(HR)', fontsize=10)
        axes[col].axis('off')

    plt.tight_layout()
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    save_path = os.path.join(OUTPUT_DIR, 'sr_comparison.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")

    return metrics


def test_denoising(pipelines: dict, clean_image_path: str):
    """
    Test denoising model.
    Takes a clean image, adds noise synthetically,
    then runs the denoising model and shows before/after.
    """
    print(f"\n--- Denoising Test ---")
    print(f"  Input: {clean_image_path}")

    if 'denoising' not in pipelines:
        print("  [SKIP] Denoising model not loaded.")
        return {}

    clean_image = Image.open(clean_image_path).convert('RGB')
    clean_array = np.array(clean_image).astype(np.float32)

    # Create three levels of degradation
    degraded_versions = {}

    # Mild Gaussian noise
    noise_mild = np.random.normal(0, 15, clean_array.shape)
    degraded_versions['Mild Noise\n(σ=15)'] = Image.fromarray(
        np.clip(clean_array + noise_mild, 0, 255).astype(np.uint8)
    )

    # Heavy Gaussian noise
    noise_heavy = np.random.normal(0, 40, clean_array.shape)
    degraded_versions['Heavy Noise\n(σ=40)'] = Image.fromarray(
        np.clip(clean_array + noise_heavy, 0, 255).astype(np.uint8)
    )

    # JPEG artifacts
    import io
    buffer = io.BytesIO()
    clean_image.save(buffer, format='JPEG', quality=10)
    buffer.seek(0)
    degraded_versions['JPEG Artifacts\n(quality=10)'] = Image.open(buffer).convert('RGB')

    # Run denoising on each degraded version
    denoised_results = {}
    metrics = {}

    for deg_name, deg_image in degraded_versions.items():
        result = pipelines['denoising'].run(
            input_image=deg_image,
            target_image=clean_image,
        )
        denoised_results[deg_name] = result['output']
        if 'psnr' in result:
            metrics[deg_name] = {'psnr': result['psnr'], 'ssim': result['ssim']}

    # Build comparison figure: for each degradation type show clean/noisy/denoised
    num_types = len(degraded_versions)
    fig, axes = plt.subplots(num_types, 3, figsize=(12, 4 * num_types))
    fig.suptitle('Denoising Results', fontsize=14, fontweight='bold')

    if num_types == 1:
        axes = [axes]

    for row, (deg_name, deg_image) in enumerate(degraded_versions.items()):
        # Clean original
        axes[row][0].imshow(clean_image)
        axes[row][0].set_title('Clean Original', fontsize=10)
        axes[row][0].axis('off')

        # Degraded input
        axes[row][1].imshow(deg_image)
        axes[row][1].set_title(f'Degraded\n{deg_name}', fontsize=10)
        axes[row][1].axis('off')

        # Denoised output
        axes[row][2].imshow(denoised_results[deg_name])
        title = 'Denoised Output'
        if deg_name in metrics:
            title += f"\nPSNR: {metrics[deg_name]['psnr']:.2f} dB"
            title += f"\nSSIM: {metrics[deg_name]['ssim']:.4f}"
        axes[row][2].set_title(title, fontsize=10)
        axes[row][2].axis('off')

    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, 'denoising_comparison.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")

    return metrics


def test_chain_pipeline(pipelines: dict, noisy_lr_image_path: str):
    """
    Test the full restoration pipeline:
    Noisy LR image → Denoise → Super-Resolve

    This is the most impressive demo — shows both models working together.
    """
    print(f"\n--- Chain Pipeline Test (Denoise → SR) ---")

    if 'denoising' not in pipelines or 'srresnet' not in pipelines:
        print("  [SKIP] Need both denoising and srresnet models.")
        return

    original = Image.open(noisy_lr_image_path).convert('RGB')
    original_array = np.array(original).astype(np.float32)

    # Create a noisy LR version
    noise = np.random.normal(0, 25, original_array.shape)
    noisy = Image.fromarray(
        np.clip(original_array + noise, 0, 255).astype(np.uint8)
    )

    # Step 1: Denoise
    print("  Step 1: Denoising...")
    denoised_result = pipelines['denoising'].run(input_image=noisy)
    denoised = denoised_result['output']

    # Step 2: Super-resolve the denoised image
    print("  Step 2: Super-resolving...")
    sr_result = pipelines['srresnet'].run(input_image=denoised)
    sr_output = sr_result['output']

    # Also run SR directly on noisy (to show chain is better)
    sr_noisy_result = pipelines['srresnet'].run(input_image=noisy)
    sr_noisy = sr_noisy_result['output']

    # Build comparison
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    fig.suptitle('Full Restoration Pipeline: Denoise → Super-Resolve', fontsize=13, fontweight='bold')

    axes[0].imshow(noisy)
    axes[0].set_title('Noisy LR Input', fontsize=11)
    axes[0].axis('off')

    axes[1].imshow(denoised)
    axes[1].set_title('After Denoising\n(same size)', fontsize=11)
    axes[1].axis('off')

    axes[2].imshow(sr_noisy)
    axes[2].set_title('SR only\n(no denoising first)', fontsize=11)
    axes[2].axis('off')

    axes[3].imshow(sr_output)
    axes[3].set_title('Denoise → SR\n(full pipeline)', fontsize=11)
    axes[3].axis('off')

    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, 'chain_pipeline.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


def print_final_table(sr_metrics: dict, denoise_metrics: dict):
    """Print a clean summary table of all results."""
    print(f"\n{'='*55}")
    print(f"  FINAL RESULTS SUMMARY")
    print(f"{'='*55}")

    print(f"\n  Super-Resolution Models (on DIV2K validation):")
    print(f"  {'Model':<15} {'PSNR':>10} {'SSIM':>10}")
    print(f"  {'-'*35}")
    for model, m in sr_metrics.items():
        print(f"  {model.upper():<15} {m['psnr']:>9.2f} dB {m['ssim']:>9.4f}")

    if denoise_metrics:
        print(f"\n  Denoising Model:")
        print(f"  {'Degradation':<25} {'PSNR':>10} {'SSIM':>10}")
        print(f"  {'-'*45}")
        for deg, m in denoise_metrics.items():
            clean_name = deg.replace('\n', ' ')
            print(f"  {clean_name:<25} {m['psnr']:>9.2f} dB {m['ssim']:>9.4f}")

    print(f"\n  All comparison images saved to: {OUTPUT_DIR}/")
    print(f"{'='*55}")


def get_div2k_test_pair(config_path: str = 'configs/config.yaml') -> tuple:
    """Get the first LR/HR pair from DIV2K validation set."""
    import yaml
    with open(config_path) as f:
        config = yaml.safe_load(f)

    data_root = config['data']['root']
    hr_dir = os.path.join(data_root, config['data']['valid_hr'])
    lr_dir = os.path.join(data_root, config['data']['valid_lr'])
    scale = config['data']['scale_factor']

    hr_files = sorted([
        f for f in os.listdir(hr_dir)
        if f.lower().endswith(('.png', '.jpg'))
    ])

    if not hr_files:
        return None, None

    hr_file = hr_files[0]
    name, ext = os.path.splitext(hr_file)
    lr_file = f"{name}x{scale}{ext}"

    hr_path = os.path.join(hr_dir, hr_file)
    lr_path = os.path.join(lr_dir, lr_file)

    if os.path.exists(lr_path):
        return lr_path, hr_path
    return None, None


def main():
    parser = argparse.ArgumentParser(description="Test all trained models")
    parser.add_argument('--image', type=str, default=None,
                        help='Path to your own test image')
    parser.add_argument('--use_div2k', action='store_true',
                        help='Use first DIV2K validation image as test')
    args = parser.parse_args()

    print(f"\n{'='*55}")
    print(f"  COMPREHENSIVE MODEL VERIFICATION")
    print(f"{'='*55}\n")

    # Load all models
    print("Loading models...")
    pipelines = load_all_pipelines()

    if not pipelines:
        print("No models loaded. Check your checkpoint paths.")
        return

    print(f"\n  {len(pipelines)} model(s) loaded: {list(pipelines.keys())}")

    # Determine test images
    if args.use_div2k or args.image is None:
        print("\nUsing DIV2K validation images for testing...")
        lr_path, hr_path = get_div2k_test_pair()
        if lr_path is None:
            print("Could not find DIV2K validation images.")
            return
        test_image_path = hr_path  # Use HR for denoising test
        print(f"  LR: {lr_path}")
        print(f"  HR: {hr_path}")
    else:
        lr_path = args.image
        hr_path = None
        test_image_path = args.image
        print(f"  Using your image: {args.image}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Run all tests
    sr_metrics = test_super_resolution(pipelines, lr_path, hr_path)
    denoise_metrics = test_denoising(pipelines, test_image_path)
    test_chain_pipeline(pipelines, lr_path)

    # Print final summary
    print_final_table(sr_metrics, denoise_metrics)


if __name__ == "__main__":
    main()