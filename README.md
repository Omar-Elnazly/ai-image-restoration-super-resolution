# AI-Based Image Restoration and Super-Resolution

Deep learning project for image super-resolution and denoising using PyTorch.

## Setup

```bash
conda env create -f environment.yml
conda activate sr_project
```

## Dataset

Download DIV2K from https://data.vision.ee.ethz.ch/cvl/DIV2K/
Place in data/DIV2K/ following the structure in config.yaml.

## Training

```bash
# Phase 1: Train SRCNN
python training/train_srcnn.py --config configs/config.yaml

# Resume training
python training/train_srcnn.py --resume

# Phase 2: Train Denoising Autoencoder
python training/train_denoising.py

# Phase 3: Train SRResNet
python training/train_srresnet.py

# Phase 4: Train SRGAN (initialize from SRResNet checkpoint)
python training/train_srgan.py --generator_pretrained checkpoints/srresnet/srresnet_best.pth
```

## Evaluation

```bash
python evaluate.py --model srcnn --checkpoint checkpoints/srcnn/srcnn_best.pth
```

## Inference

```bash
# Run Gradio UI
python ui/app.py
```

## TensorBoard

```bash
tensorboard --logdir logs/
# Open http://localhost:6006
```
