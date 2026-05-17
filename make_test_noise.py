"""
Creates test images for denoising demonstration.
Run: python make_test_noise.py
"""
import numpy as np
from PIL import Image
import io
import os

os.makedirs("test_images", exist_ok=True)

# Use any image you have — change this path
source_path = "data/DIV2K/DIV2K_valid_HR/0801.png"
img = Image.open(source_path).convert("RGB")

# Crop a nice 512x512 region for clean demo
img = img.crop((0, 0, 512, 512))
img.save("test_images/clean_original.png")
print("Saved: test_images/clean_original.png")

arr = np.array(img).astype(np.float32)

# Test image 1 — Mild Gaussian noise
noise_mild = np.random.normal(0, 20, arr.shape)
noisy_mild = np.clip(arr + noise_mild, 0, 255).astype(np.uint8)
Image.fromarray(noisy_mild).save("test_images/noisy_mild.png")
print("Saved: test_images/noisy_mild.png")

# Test image 2 — Heavy Gaussian noise
noise_heavy = np.random.normal(0, 50, arr.shape)
noisy_heavy = np.clip(arr + noise_heavy, 0, 255).astype(np.uint8)
Image.fromarray(noisy_heavy).save("test_images/noisy_heavy.png")
print("Saved: test_images/noisy_heavy.png")

# Test image 3 — JPEG compression artifacts
buffer = io.BytesIO()
img.save(buffer, format='JPEG', quality=5)
buffer.seek(0)
jpeg_degraded = Image.open(buffer).convert("RGB")
jpeg_degraded.save("test_images/jpeg_artifacts.png")
print("Saved: test_images/jpeg_artifacts.png")

print("\nDone. Upload these to the UI and run Denoising Autoencoder.")