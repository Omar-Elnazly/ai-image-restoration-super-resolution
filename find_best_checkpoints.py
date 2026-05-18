"""
Find Best Checkpoints (Combined + Original)

Evaluates ALL saved checkpoints for each model across both
  - checkpoints_combined/<model>/   (combined training runs)
  - checkpoints/<model>/            (original DIV2K runs)
and reports which one achieves the highest PSNR.

Run:
    python find_best_checkpoints.py

This tells you exactly which checkpoint file to use
in app.py for each model.
"""

import os
from typing import Optional
import sys
import glob
import yaml
import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from datasets.div2k_dataset import DIV2KDataset
from models.srcnn import build_srcnn
from models.autoencoder import DenoisingAutoencoder
from models.srresnet import SRResNet
from datasets.denoising_dataset import DenoisingDataset
from utils.metrics import calculate_psnr, calculate_ssim, MetricsTracker
from utils.checkpoint import load_checkpoint


# ----------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------
CONFIG_PATH     = "configs/config.yaml"
COMBINED_DIR    = "checkpoints_combined"   # FIX: was "DIVK2_DIR" (typo)
DIV2K_DIR       = "checkpoints"
SCALE           = 2
MAX_EVAL_IMAGES = 30   # Use 30 images for speed — enough to compare

# Fixed seed for reproducible noise in denoising evaluation.
# Without this, each checkpoint sees different random noise,
# making PSNR comparisons unfair.
EVAL_SEED = 42


def get_device():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")
    return device


def get_val_loader_sr(config: dict):
    """Validation loader for super-resolution models."""
    data_cfg  = config['data']
    patch_cfg = config['patches']

    dataset = DIV2KDataset(
        hr_dir=os.path.join(data_cfg['root'], data_cfg['valid_hr']),
        lr_dir=os.path.join(data_cfg['root'], data_cfg['valid_lr']),
        hr_patch_size=patch_cfg['hr_patch_size'],
        scale=data_cfg['scale_factor'],
        max_images=MAX_EVAL_IMAGES,
        augment=False,
        split='val',
    )

    return DataLoader(
        dataset,
        batch_size=4,
        shuffle=False,
        num_workers=0,
        pin_memory=(torch.cuda.is_available()),
    )


def get_val_loader_denoising(config: dict):
    """Validation loader for denoising model."""
    data_cfg  = config['data']
    patch_cfg = config['patches']

    dataset = DenoisingDataset(
        hr_dir=os.path.join(data_cfg['root'], data_cfg['valid_hr']),
        patch_size=patch_cfg['hr_patch_size'],
        max_images=MAX_EVAL_IMAGES,
        noise_type='mixed',
        noise_level=0.1,
        jpeg_quality=30,
        augment=False,
    )

    return DataLoader(
        dataset,
        batch_size=4,
        shuffle=False,
        num_workers=0,
        pin_memory=(torch.cuda.is_available()),
    )


@torch.no_grad()
def evaluate_sr_checkpoint(checkpoint_path: str, model: torch.nn.Module,
                             val_loader, device: torch.device,
                             needs_bicubic: bool = False) -> float:
    """Evaluate a single SR checkpoint and return average PSNR."""
    try:
        load_checkpoint(checkpoint_path, model, device=device)
    except Exception as e:
        print(f"    Could not load: {e}")
        return 0.0

    model.eval()
    tracker = MetricsTracker()

    for lr_batch, hr_batch in val_loader:
        lr_batch = lr_batch.to(device)
        hr_batch = hr_batch.to(device)

        if needs_bicubic:
            # SRCNN needs bicubic upsampled input
            model_input = torch.nn.functional.interpolate(
                lr_batch, scale_factor=SCALE,
                mode='bicubic', align_corners=False
            )
        else:
            model_input = lr_batch

        # FIX: use torch.amp.autocast instead of deprecated torch.cuda.amp.autocast
        with torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):
            pred = model(model_input)

        psnr = calculate_psnr(pred, hr_batch)
        # FIX: weight by hr_batch size (semantically correct — PSNR is on HR images)
        tracker.update('psnr', psnr, hr_batch.size(0))

    return tracker.average('psnr')


@torch.no_grad()
def evaluate_denoising_checkpoint(checkpoint_path: str, model: torch.nn.Module,
                                   val_loader, device: torch.device) -> float:
    """Evaluate a single denoising checkpoint and return average PSNR.

    FIX: Seeds are set before every evaluation so every checkpoint sees
    identical noise — without this, PSNR comparisons are unfair because
    each checkpoint would be evaluated against different random noise.
    """
    try:
        load_checkpoint(checkpoint_path, model, device=device)
    except Exception as e:
        print(f"    Could not load: {e}")
        return 0.0

    model.eval()
    tracker = MetricsTracker()

    # FIX: reset RNG to the same state before every checkpoint evaluation
    # so all checkpoints see exactly the same noisy images.
    torch.manual_seed(EVAL_SEED)
    np.random.seed(EVAL_SEED)

    for noisy_batch, clean_batch in val_loader:
        noisy_batch = noisy_batch.to(device)
        clean_batch = clean_batch.to(device)

        # FIX: use torch.amp.autocast instead of deprecated torch.cuda.amp.autocast
        with torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):
            pred = model(noisy_batch)

        psnr = calculate_psnr(pred, clean_batch)
        tracker.update('psnr', psnr, noisy_batch.size(0))

    return tracker.average('psnr')


def find_checkpoints(directory: str, prefix: str) -> list:
    """Find all checkpoint files matching a prefix in a directory."""
    pattern = os.path.join(directory, f"{prefix}*.pth")
    files   = sorted(glob.glob(pattern))
    return files


def evaluate_all_checkpoints_for_model(
    model_name: str,
    checkpoint_dirs: list,   # list of (ckpt_dir, prefix, source_tag) 3-tuples
    model,
    val_loader,
    device,
    evaluate_fn,
    **eval_kwargs,
) -> dict:
    """
    Evaluate all checkpoints for one model across multiple directories
    and return results keyed by '<filename> [<source_tag>]'.

    checkpoint_dirs entries are 3-tuples:
        (ckpt_dir, prefix, source_tag)

    source_tag is the human-readable label stored in result keys so we
    can reconstruct the exact path later without any string guessing.
    It should be one of COMBINED_DIR or DIV2K_DIR.
    """
    print(f"\n{'─'*55}")
    print(f"  Evaluating: {model_name}")

    # Collect every checkpoint file from every (dir, prefix, tag) triple
    all_checkpoints = []
    for ckpt_dir, prefix, source_tag in checkpoint_dirs:
        found = find_checkpoints(ckpt_dir, prefix)
        print(f"  Directory:  {ckpt_dir}  →  {len(found)} file(s)")
        for path in found:
            all_checkpoints.append((path, ckpt_dir, source_tag))

    print(f"{'─'*55}")

    if not all_checkpoints:
        print(f"  No checkpoints found in any directory.")
        return {}

    print(f"  Total checkpoints to evaluate: {len(all_checkpoints)}")

    results = {}

    for ckpt_path, ckpt_dir, source_tag in tqdm(all_checkpoints, desc=f"  {model_name}", leave=False):
        filename = os.path.basename(ckpt_path)

        # FIX: source_tag is now an explicit parameter passed in by the
        # caller (COMBINED_DIR or DIV2K_DIR), so it is always correct
        # regardless of whether relative or absolute paths are used.
        result_key = f"{filename} [{source_tag}]"
        psnr = evaluate_fn(ckpt_path, model, val_loader, device, **eval_kwargs)
        results[result_key] = psnr
        tqdm.write(f"    {result_key:<60} PSNR: {psnr:.2f} dB")

    return results


def resolve_checkpoint_path(result_key: str, subdir: str) -> Optional[str]:
    """
    Reconstruct the full path for a result_key of the form
    'filename.pth [source_tag]', where source_tag is the base
    checkpoint directory (e.g. 'checkpoints_combined').

    FIX: Previously the code did a brute-force search across all
    (base_dir, subdir) combinations and took the first os.path.exists()
    hit, ignoring the source_tag entirely. This meant that if the same
    filename existed in both directories, the wrong one could be
    returned silently. Now we decode the source_tag directly and only
    look in the directory that actually produced the best PSNR.
    """
    if " [" not in result_key:
        return None

    filename, tag_part = result_key.rsplit(" [", 1)
    source_tag = tag_part.rstrip("]")

    candidate = os.path.join(source_tag, subdir, filename)
    if os.path.exists(candidate):
        return candidate.replace("\\", "/")

    return None


def print_best_summary(all_results: dict, model_subdirs: dict):
    """Print a clean summary of the best checkpoint for each model.

    model_subdirs maps model_name → subdir string, e.g.:
        {'SRCNN': 'srcnn', 'Denoising': 'denoising', ...}
    """
    print(f"\n{'='*60}")
    print(f"  BEST CHECKPOINTS SUMMARY")
    print(f"{'='*60}")

    for model_name, results in all_results.items():
        if not results:
            print(f"\n  {model_name}: No checkpoints found")
            continue

        # Sort by PSNR descending
        sorted_results = sorted(results.items(), key=lambda x: x[1], reverse=True)
        best_file, best_psnr = sorted_results[0]

        subdir    = model_subdirs[model_name]
        # FIX: use resolve_checkpoint_path which honours the source_tag
        best_path = resolve_checkpoint_path(best_file, subdir)

        print(f"\n  {model_name}")
        print(f"    Best file : {best_file}")
        print(f"    Best PSNR : {best_psnr:.2f} dB")
        print(f"    Full path : {best_path}")

        # Top 3
        print(f"    Top 3:")
        for i, (fname, psnr) in enumerate(sorted_results[:3]):
            marker = "← BEST" if i == 0 else ""
            print(f"      {i+1}. {fname:<45} {psnr:.2f} dB  {marker}")

    # Print app.py update instructions
    print(f"\n{'='*60}")
    print(f"  UPDATE app.py — AVAILABLE_MODELS CHECKPOINTS")
    print(f"{'='*60}")
    print(f"""
  Replace the checkpoint paths in ui/app.py:

  AVAILABLE_MODELS = {{
      "SRCNN (Super-Resolution x2)": {{
          "type": "srcnn",
          "checkpoint": "{get_best_path(all_results, 'SRCNN', model_subdirs['SRCNN'])}",
      }},
      "Denoising Autoencoder": {{
          "type": "denoising",
          "checkpoint": "{get_best_path(all_results, 'Denoising', model_subdirs['Denoising'])}",
      }},
      "SRResNet (Super-Resolution x2)": {{
          "type": "srresnet",
          "checkpoint": "{get_best_path(all_results, 'SRResNet', model_subdirs['SRResNet'])}",
      }},
      "SRGAN (Super-Resolution x2)": {{
          "type": "srgan",
          "checkpoint": "{get_best_path(all_results, 'SRGAN', model_subdirs['SRGAN'])}",
      }},
  }}
""")


def get_best_path(all_results: dict, model_key: str, subdir: str) -> str:
    """Get the full path of the best checkpoint for a model.

    FIX: Previously this stripped the source_tag from the result key
    and then tried COMBINED_DIR first regardless — meaning if the same
    filename existed in both directories, the wrong weights could be
    written into app.py silently. Now we use resolve_checkpoint_path
    which honours the source_tag embedded in the result key.
    """
    for name, results in all_results.items():
        if model_key.lower() in name.lower() and results:
            best_key = max(results.items(), key=lambda x: x[1])[0]
            path = resolve_checkpoint_path(best_key, subdir)
            if path:
                return path
    # Fallback placeholder — callers should treat this as a warning signal
    return f"{COMBINED_DIR}/{subdir}/best.pth"


def main():
    print(f"\n{'='*60}")
    print(f"  BEST CHECKPOINT FINDER")
    print(f"  Evaluating checkpoints from: {COMBINED_DIR}/ + {DIV2K_DIR}/")
    print(f"{'='*60}\n")

    # Load config
    with open(CONFIG_PATH) as f:
        config = yaml.safe_load(f)

    device = get_device()

    # Build dataloaders
    print("\n  Building validation dataloaders...")
    val_loader_sr        = get_val_loader_sr(config)
    val_loader_denoising = get_val_loader_denoising(config)
    print(f"  SR val loader       : {len(val_loader_sr.dataset)} images")
    print(f"  Denoising val loader: {len(val_loader_denoising.dataset)} images")

    all_results = {}

    # Maps each model_name to the subdirectory it lives in under each base dir.
    # Used by print_best_summary and get_best_path to resolve full paths.
    model_subdirs = {
        'SRCNN':     'srcnn',
        'Denoising': 'denoising',
        'SRResNet':  'srresnet',
        'SRGAN':     'srgan',
    }

    # ----------------------------------------------------------------
    # SRCNN — evaluate checkpoints from both folders
    # ----------------------------------------------------------------
    srcnn_model = build_srcnn(config).to(device)
    srcnn_results = evaluate_all_checkpoints_for_model(
        model_name="SRCNN",
        # FIX: 3-tuples (ckpt_dir, prefix, source_tag).
        # source_tag is always the top-level base dir constant so it is
        # stable regardless of cwd or absolute/relative path differences.
        checkpoint_dirs=[
            (os.path.join(COMBINED_DIR, 'srcnn'), 'srcnn_combined', COMBINED_DIR),
            (os.path.join(DIV2K_DIR,   'srcnn'), 'srcnn',          DIV2K_DIR),
        ],
        model=srcnn_model,
        val_loader=val_loader_sr,
        device=device,
        evaluate_fn=evaluate_sr_checkpoint,
        needs_bicubic=True,
    )

    all_results['SRCNN'] = srcnn_results
    del srcnn_model
    torch.cuda.empty_cache()

    # ----------------------------------------------------------------
    # Denoising Autoencoder — evaluate checkpoints from both folders
    # ----------------------------------------------------------------
    denoising_model = DenoisingAutoencoder(in_channels=3, base_filters=32).to(device)
    denoising_results = evaluate_all_checkpoints_for_model(
        model_name="Denoising",
        checkpoint_dirs=[
            (os.path.join(COMBINED_DIR, 'denoising'), 'denoising_combined', COMBINED_DIR),
            (os.path.join(DIV2K_DIR,   'denoising'), 'denoising',          DIV2K_DIR),
        ],
        model=denoising_model,
        val_loader=val_loader_denoising,
        device=device,
        evaluate_fn=evaluate_denoising_checkpoint,
    )

    all_results['Denoising'] = denoising_results
    del denoising_model
    torch.cuda.empty_cache()

    # ----------------------------------------------------------------
    # SRResNet — evaluate checkpoints from both folders
    # ----------------------------------------------------------------
    srresnet_model = SRResNet(
        scale_factor=SCALE, num_channels=3,
        num_filters=64, num_res_blocks=16
    ).to(device)

    srresnet_results = evaluate_all_checkpoints_for_model(
        model_name="SRResNet",
        checkpoint_dirs=[
            (os.path.join(COMBINED_DIR, 'srresnet'), 'srresnet_combined', COMBINED_DIR),
            (os.path.join(DIV2K_DIR,   'srresnet'), 'srresnet',          DIV2K_DIR),
        ],
        model=srresnet_model,
        val_loader=val_loader_sr,
        device=device,
        evaluate_fn=evaluate_sr_checkpoint,
        needs_bicubic=False,
    )

    all_results['SRResNet'] = srresnet_results
    del srresnet_model
    torch.cuda.empty_cache()

    # ----------------------------------------------------------------
    # SRGAN — evaluate checkpoints from both folders
    # ----------------------------------------------------------------
    srgan_model = SRResNet(
        scale_factor=SCALE, num_channels=3,
        num_filters=64, num_res_blocks=16
    ).to(device)

    srgan_results = evaluate_all_checkpoints_for_model(
        model_name="SRGAN",
        checkpoint_dirs=[
            (os.path.join(COMBINED_DIR, 'srgan'), 'srgan_combined_gen', COMBINED_DIR),
            (os.path.join(DIV2K_DIR,   'srgan'), 'srgan_gen',          DIV2K_DIR),
        ],
        model=srgan_model,
        val_loader=val_loader_sr,
        device=device,
        evaluate_fn=evaluate_sr_checkpoint,
        needs_bicubic=False,
    )

    all_results['SRGAN'] = srgan_results
    del srgan_model
    torch.cuda.empty_cache()

    # ----------------------------------------------------------------
    # Print final summary
    # ----------------------------------------------------------------
    print_best_summary(all_results, model_subdirs)

    # Save results to file for reference
    results_path = "best_checkpoints_report.txt"
    with open(results_path, 'w') as f:
        f.write("BEST CHECKPOINTS REPORT\n")
        f.write("="*60 + "\n\n")
        for model_name, results in all_results.items():
            if results:
                sorted_r = sorted(results.items(), key=lambda x: x[1], reverse=True)
                f.write(f"{model_name}\n")
                for fname, psnr in sorted_r:
                    marker = " <- BEST" if fname == sorted_r[0][0] else ""
                    f.write(f"  {fname:<50} {psnr:.2f} dB{marker}\n")
                f.write("\n")

    print(f"\n  Full report saved to: {results_path}")
    print(f"  Use the checkpoint paths shown above to update ui/app.py\n")


if __name__ == "__main__":
    main()