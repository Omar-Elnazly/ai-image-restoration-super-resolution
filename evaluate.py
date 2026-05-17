"""
Standalone evaluation script.

Evaluates a trained model on the full validation set and reports:
  - Average PSNR
  - Average SSIM
  - Per-image metrics (optional)

Usage:
    python evaluate.py --model srcnn --checkpoint checkpoints/srcnn/srcnn_best.pth
    python evaluate.py --model srresnet --checkpoint checkpoints/srresnet/srresnet_best.pth
"""

import os
import sys
import argparse
import yaml
import torch
from PIL import Image
import torchvision.transforms.functional as TF
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from inference.inference_pipeline import InferencePipeline
from utils.metrics import calculate_psnr, calculate_ssim, MetricsTracker


def evaluate(model_type: str, checkpoint_path: str, config_path: str = "configs/config.yaml"):

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    data_cfg = config['data']
    data_root = data_cfg['root']

    # Load pipeline
    pipeline = InferencePipeline(
        model_type=model_type,
        checkpoint_path=checkpoint_path,
        config_path=config_path,
        device='auto',
    )

    # Collect validation image pairs
    hr_dir = os.path.join(data_root, data_cfg['valid_hr'])
    lr_dir = os.path.join(data_root, data_cfg['valid_lr'])
    scale = data_cfg['scale_factor']
    max_images = data_cfg['max_valid_images']

    hr_files = sorted([
        f for f in os.listdir(hr_dir)
        if f.lower().endswith(('.png', '.jpg', '.jpeg'))
    ])

    if max_images:
        hr_files = hr_files[:max_images]

    tracker = MetricsTracker()
    per_image_results = []

    print(f"\nEvaluating {model_type} on {len(hr_files)} validation images...\n")

    for hr_file in tqdm(hr_files):
        hr_path = os.path.join(hr_dir, hr_file)

        name, ext = os.path.splitext(hr_file)
        lr_file = f"{name}x{scale}{ext}"
        lr_path = os.path.join(lr_dir, lr_file)

        if not os.path.exists(lr_path):
            continue

        result = pipeline.run(lr_path, target_image=hr_path)

        if 'psnr' in result:
            tracker.update('psnr', result['psnr'])
            tracker.update('ssim', result['ssim'])
            per_image_results.append({
                'file': hr_file,
                'psnr': result['psnr'],
                'ssim': result['ssim'],
            })

    # Print summary
    avg = tracker.summary()
    print(f"\n{'='*40}")
    print(f"  Model:     {model_type}")
    print(f"  Images:    {len(per_image_results)}")
    print(f"  Avg PSNR:  {avg.get('psnr', 0):.2f} dB")
    print(f"  Avg SSIM:  {avg.get('ssim', 0):.4f}")
    print(f"{'='*40}")

    # Print worst and best cases
    if per_image_results:
        best = max(per_image_results, key=lambda x: x['psnr'])
        worst = min(per_image_results, key=lambda x: x['psnr'])
        print(f"\n  Best:  {best['file']} — PSNR {best['psnr']:.2f} dB")
        print(f"  Worst: {worst['file']} — PSNR {worst['psnr']:.2f} dB")

    return avg


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate trained model")
    parser.add_argument('--model', type=str, required=True,
                        choices=['srcnn', 'denoising', 'srresnet', 'srgan'],
                        help='Model type to evaluate')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint (.pth file)')
    parser.add_argument('--config', type=str, default='configs/config.yaml')
    args = parser.parse_args()

    evaluate(args.model, args.checkpoint, args.config)