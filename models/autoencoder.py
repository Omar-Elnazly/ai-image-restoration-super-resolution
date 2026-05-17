"""
Denoising Convolutional Autoencoder.

Architecture: Encoder → Bottleneck → Decoder (with skip connections)

The encoder compresses the image into a compact feature representation.
The decoder reconstructs the clean image from those features.
Skip connections (like U-Net) help preserve spatial details.

Input:  Noisy/degraded image (C, H, W)
Output: Clean reconstructed image (C, H, W)

The model does NOT change spatial resolution —
output is same size as input.
"""

import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    """
    Basic building block: Conv → BatchNorm → ReLU → Conv → BatchNorm → ReLU
    
    BatchNorm helps training stability and reduces internal covariate shift.
    Using two convolutions per block increases receptive field.
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class DenoisingAutoencoder(nn.Module):
    """
    U-Net style denoising autoencoder.
    
    Encoder: 3 → 32 → 64 → 128 (with max pooling)
    Bottleneck: 128 → 256 → 128
    Decoder: 128+128 → 64 → 64+64 → 32 → 32+32 → 3
             (+ denotes skip connection concatenation)
    
    Why skip connections?
    Without them, fine spatial details (edges, textures) can get lost
    in the bottleneck. Skips let the decoder access them directly.
    """

    def __init__(self, in_channels: int = 3, base_filters: int = 32):
        super().__init__()

        f = base_filters  # 32

        # ---- Encoder ----
        self.enc1 = ConvBlock(in_channels, f)        # (B, 32, H, W)
        self.pool1 = nn.MaxPool2d(2)                 # (B, 32, H/2, W/2)

        self.enc2 = ConvBlock(f, f * 2)              # (B, 64, H/2, W/2)
        self.pool2 = nn.MaxPool2d(2)                 # (B, 64, H/4, W/4)

        self.enc3 = ConvBlock(f * 2, f * 4)          # (B, 128, H/4, W/4)
        self.pool3 = nn.MaxPool2d(2)                 # (B, 128, H/8, W/8)

        # ---- Bottleneck ----
        self.bottleneck = ConvBlock(f * 4, f * 8)    # (B, 256, H/8, W/8)

        # ---- Decoder ----
        # Upsample + concatenate skip connection + convblock
        self.up3 = nn.ConvTranspose2d(f * 8, f * 4, kernel_size=2, stride=2)  # (B, 128, H/4, W/4)
        self.dec3 = ConvBlock(f * 8, f * 4)          # f*4 + f*4 from skip = f*8 input

        self.up2 = nn.ConvTranspose2d(f * 4, f * 2, kernel_size=2, stride=2)  # (B, 64, H/2, W/2)
        self.dec2 = ConvBlock(f * 4, f * 2)          # f*2 + f*2 from skip = f*4 input

        self.up1 = nn.ConvTranspose2d(f * 2, f, kernel_size=2, stride=2)      # (B, 32, H, W)
        self.dec1 = ConvBlock(f * 2, f)              # f + f from skip = f*2 input

        # Output layer: map to RGB, no activation (sigmoid applied at loss or clamp at inference)
        self.output_conv = nn.Conv2d(f, in_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Degraded image, shape (B, C, H, W), [0, 1]
               H and W must be divisible by 8 (due to 3 pooling layers)
        Returns:
            Restored image, same shape as input
        """
        # Encoder
        e1 = self.enc1(x)     # Save for skip
        e2 = self.enc2(self.pool1(e1))   # Save for skip
        e3 = self.enc3(self.pool2(e2))   # Save for skip

        # Bottleneck
        b = self.bottleneck(self.pool3(e3))

        # Decoder with skip connections
        # Concatenate along channel dimension
        d3 = self.up3(b)
        d3 = self.dec3(torch.cat([d3, e3], dim=1))

        d2 = self.up2(d3)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))

        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))

        out = self.output_conv(d1)

        # Use sigmoid to keep output in [0, 1]
        return torch.sigmoid(out)