"""
SRGAN — Super-Resolution Generative Adversarial Network
Paper: Ledig et al., CVPR 2017

SRGAN adds a discriminator and perceptual loss on top of SRResNet.

Generator: SRResNet (same as before)
Discriminator: VGG-like network that classifies real HR vs fake SR

Loss function:
  Total = Content Loss + 0.001 × Adversarial Loss
  
  Content Loss: MSE on VGG19 feature maps (perceptual loss)
                Measures perceptual similarity, not just pixel accuracy
  
  Adversarial Loss: Generator loss from discriminator
                    Pushes generator to produce "real-looking" images

Why perceptual loss?
  Pixel-wise MSE tends to produce blurry images (average of plausible solutions).
  Perceptual loss in feature space encourages sharper, more realistic textures.
"""

import torch
import torch.nn as nn
import torchvision.models as tv_models


class DiscriminatorBlock(nn.Module):
    """Basic discriminator block: Conv → BN → LeakyReLU."""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class SRGANDiscriminator(nn.Module):
    """
    VGG-like discriminator for SRGAN.
    
    Outputs a probability that the input image is real (not SR generated).
    Uses strided convolutions to reduce spatial size progressively.
    
    Input: HR image patch (B, 3, 128, 128)
    Output: Scalar probability per batch item
    """

    def __init__(self, in_channels: int = 3):
        super().__init__()

        # First layer has no BN (common practice for discriminators)
        self.first = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # Stacked discriminator blocks
        # Spatial size decreases with stride=2 blocks
        self.features = nn.Sequential(
            DiscriminatorBlock(64,  64,  stride=2),   # /2
            DiscriminatorBlock(64,  128, stride=1),
            DiscriminatorBlock(128, 128, stride=2),   # /4
            DiscriminatorBlock(128, 256, stride=1),
            DiscriminatorBlock(256, 256, stride=2),   # /8
            DiscriminatorBlock(256, 512, stride=1),
            DiscriminatorBlock(512, 512, stride=2),   # /16
        )

        # Global average pooling then classification
        # AdaptiveAvgPool2d(1) → spatial size becomes 1×1
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, 1),
            # No sigmoid — use BCEWithLogitsLoss which is numerically stable
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.first(x)
        x = self.features(x)
        return self.classifier(x)


class VGGFeatureExtractor(nn.Module):
    """
    Extracts intermediate feature maps from a pretrained VGG19.
    Used to compute perceptual (content) loss.
    
    We use features up to relu3_3 (layer index 18) by default.
    Deeper features capture more semantic content.
    
    The VGG is FROZEN — we only use it for loss computation, not training.
    """

    def __init__(self, feature_layer: int = 18, use_bn: bool = False):
        super().__init__()

        # Load pretrained VGG19
        if use_bn:
            vgg = tv_models.vgg19_bn(weights=tv_models.VGG19_BN_Weights.DEFAULT)
        else:
            vgg = tv_models.vgg19(weights=tv_models.VGG19_Weights.DEFAULT)

        # Use only up to feature_layer
        self.features = nn.Sequential(*list(vgg.features.children())[:feature_layer + 1])

        # Freeze all parameters — VGG is fixed, not trained
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.features(x)


class PerceptualLoss(nn.Module):
    """
    Combines pixel-wise loss with VGG perceptual loss.
    
    Total = pixel_weight × MSE(pred, target)
          + vgg_weight × MSE(VGG(pred), VGG(target))
    """

    def __init__(self, pixel_weight: float = 1.0, vgg_weight: float = 0.006):
        super().__init__()
        self.pixel_weight = pixel_weight
        self.vgg_weight = vgg_weight
        self.vgg = VGGFeatureExtractor()
        self.mse = nn.MSELoss()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # Pixel loss
        pixel_loss = self.mse(pred, target)

        # Perceptual loss (VGG features)
        pred_features = self.vgg(pred)
        target_features = self.vgg(target.detach())  # detach target — no gradients needed
        vgg_loss = self.mse(pred_features, target_features)

        return self.pixel_weight * pixel_loss + self.vgg_weight * vgg_loss