"""
SRResNet — Super-Resolution Residual Network
From: "Photo-Realistic Single Image Super-Resolution Using a GAN" (SRGAN paper)
      Ledig et al., CVPR 2017

Key improvements over SRCNN:
  1. Much deeper network (16 residual blocks)
  2. Sub-pixel convolution for upsampling (PixelShuffle)
  3. Residual learning — easier to train deep networks
  4. Takes RAW LR image as input (not bicubic upscaled)

Architecture:
  LR input → Conv → [Residual Block x 16] → Conv → PixelShuffle x2 → Conv → HR output

PixelShuffle (sub-pixel convolution):
  Instead of transposed convolution, we use PixelShuffle:
  Output channels (r²×C) → rearranged to (C, H×r, W×r)
  This avoids checkerboard artifacts common with transposed convolutions.
"""

import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    """
    Basic residual block used in SRResNet.
    
    Structure: Conv → BN → PReLU → Conv → BN → (+ skip connection)
    
    PReLU (Parametric ReLU): learnable version of LeakyReLU.
    Better than ReLU for SR tasks as it avoids dying neurons.
    """

    def __init__(self, num_filters: int = 64):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(num_filters, num_filters, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(num_filters),
            nn.PReLU(),
            nn.Conv2d(num_filters, num_filters, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(num_filters),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Residual connection: output = F(x) + x
        return self.block(x) + x


class UpsampleBlock(nn.Module):
    """
    Sub-pixel convolution upsampling block.
    
    PixelShuffle rearranges (B, C*r², H, W) → (B, C, H*r, W*r)
    where r is the upscale factor.
    
    For r=2: we need C*4 channels going in, get C channels with doubled resolution.
    """

    def __init__(self, in_channels: int, scale_factor: int = 2):
        super().__init__()
        self.block = nn.Sequential(
            # Output r² times more channels than needed
            nn.Conv2d(in_channels, in_channels * (scale_factor ** 2), kernel_size=3, padding=1),
            nn.PixelShuffle(scale_factor),   # Rearrange to higher resolution
            nn.PReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class SRResNet(nn.Module):
    """
    SRResNet for x2 super-resolution.
    
    Args:
        scale_factor:   Upscaling factor (2 for x2)
        num_channels:   Input/output channels (3 for RGB)
        num_filters:    Feature map channels (64 default)
        num_res_blocks: Number of residual blocks (16 default, reduce to 8 for speed)
    """

    def __init__(
        self,
        scale_factor: int = 2,
        num_channels: int = 3,
        num_filters: int = 64,
        num_res_blocks: int = 16,
    ):
        super().__init__()

        # Initial feature extraction
        self.head = nn.Sequential(
            nn.Conv2d(num_channels, num_filters, kernel_size=9, padding=4),
            nn.PReLU(),
        )

        # Residual blocks (main body)
        self.body = nn.Sequential(
            *[ResidualBlock(num_filters) for _ in range(num_res_blocks)]
        )

        # Post-residual convolution (connects body back to skip connection)
        self.post_res_conv = nn.Sequential(
            nn.Conv2d(num_filters, num_filters, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(num_filters),
        )

        # Upsampling: one UpsampleBlock per scale step of 2
        # For x2: one block; for x4: two blocks
        upsample_blocks = []
        current_scale = scale_factor
        while current_scale > 1:
            upsample_blocks.append(UpsampleBlock(num_filters, scale_factor=2))
            current_scale //= 2
        self.upsample = nn.Sequential(*upsample_blocks)

        # Output convolution
        self.tail = nn.Conv2d(num_filters, num_channels, kernel_size=9, padding=4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: LR image, shape (B, C, H, W), [0, 1]
        Returns:
            SR image, shape (B, C, H*scale, W*scale), [0, 1]
        """
        head_out = self.head(x)

        # Residual learning: add skip around entire residual body
        body_out = self.post_res_conv(self.body(head_out))
        body_out = body_out + head_out  # Global residual connection

        # Upsample
        up = self.upsample(body_out)

        # Final output — tanh maps to [-1, 1]; add 0.5 to center, or use clamp
        out = self.tail(up)
        return torch.clamp(out, 0, 1)  # Clamp to valid range