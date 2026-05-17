"""
SRCNN — Super-Resolution Convolutional Neural Network
Paper: "Learning a Deep Convolutional Network for Image Super-Resolution"
       Dong et al., ECCV 2014

Architecture:
  Layer 1: Feature extraction — large 9x9 kernel, captures local patches
  Layer 2: Non-linear mapping — 1x1 kernel, maps features to SR space
  Layer 3: Reconstruction — 5x5 kernel, reconstructs HR pixels

Important detail:
  SRCNN takes a BICUBIC UPSCALED LR image as input (not the raw LR image).
  The network learns to remove bicubic artifacts rather than perform upsampling.
  This means the input and output are the SAME spatial size.

For x2 SR with patch size 64 (LR):
  - Bicubic upsample to 128x128 first
  - Feed 128x128 to SRCNN
  - Output is 128x128 (refined HR)
"""

import torch
import torch.nn as nn


class SRCNN(nn.Module):
    """
    SRCNN model.
    
    Args:
        num_channels: Number of input/output channels (3 for RGB)
        f1: Number of filters in layer 1
        f2: Number of filters in layer 2
        k1: Kernel size for layer 1 (original paper: 9)
        k2: Kernel size for layer 2 (original paper: 1)
        k3: Kernel size for layer 3 (original paper: 5)
    """

    def __init__(
        self,
        num_channels: int = 3,
        f1: int = 64,
        f2: int = 32,
        k1: int = 9,
        k2: int = 1,
        k3: int = 5,
    ):
        super(SRCNN, self).__init__()

        # Layer 1: Feature extraction
        # Large kernel (9x9) to capture large local patches
        # padding = k//2 to keep spatial dimensions the same
        self.layer1 = nn.Sequential(
            nn.Conv2d(num_channels, f1, kernel_size=k1, padding=k1 // 2),
            nn.ReLU(inplace=True)  # inplace saves a tiny bit of memory
        )

        # Layer 2: Non-linear mapping
        # 1x1 convolution acts as channel-wise fully connected layer
        self.layer2 = nn.Sequential(
            nn.Conv2d(f1, f2, kernel_size=k2, padding=k2 // 2),
            nn.ReLU(inplace=True)
        )

        # Layer 3: Reconstruction
        # Outputs same number of channels as input (RGB)
        # No activation here — we want full range output
        self.layer3 = nn.Conv2d(f2, num_channels, kernel_size=k3, padding=k3 // 2)

        # Initialize weights using He initialization (good for ReLU networks)
        self._initialize_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Bicubic-upscaled LR image, shape (B, C, H, W), values [0, 1]
               where H, W are the TARGET (HR) spatial dimensions
        
        Returns:
            Refined SR image, same shape as input
        """
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x

    def _initialize_weights(self):
        """
        He (Kaiming) initialization:
        Designed for ReLU activations — keeps variance stable across layers.
        Better than default random init for deep networks.
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)


def build_srcnn(config: dict) -> SRCNN:
    """
    Build SRCNN model from config dictionary.
    
    Args:
        config: Must contain 'srcnn' key with model parameters
    """
    cfg = config.get('srcnn', {})
    return SRCNN(
        num_channels=cfg.get('num_channels', 3),
        f1=cfg.get('f1', 64),
        f2=cfg.get('f2', 32),
        k1=cfg.get('k1', 9),
        k2=cfg.get('k2', 1),
        k3=cfg.get('k3', 5),
    )