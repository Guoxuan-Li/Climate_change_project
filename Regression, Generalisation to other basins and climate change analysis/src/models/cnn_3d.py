"""
Stage 4: CNN encoder for Data3D atmospheric fields.

Modified ResNet-18 adapted for 13-channel input.
"""
import torch
import torch.nn as nn
import torchvision.models as models

from src.config import DATA3D_CHANNELS, NUM_DIRECTION_CLASSES, HIDDEN_DIM, MLP_DROPOUT


class CNNEncoder3D(nn.Module):
    """ResNet-18 backbone adapted for 13-channel atmospheric grids.

    Input: (B, 13, 81, 81)
    Output: (B, num_classes) logits
    """

    def __init__(
        self,
        in_channels: int = DATA3D_CHANNELS,
        num_classes: int = NUM_DIRECTION_CLASSES,
        feature_dim: int = HIDDEN_DIM,
    ):
        super().__init__()
        # Load ResNet-18 without pretrained weights
        resnet = models.resnet18(weights=None)

        # Replace first conv for 13 input channels
        resnet.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2,
                                 padding=3, bias=False)

        # Remove the final FC layer, keep everything else
        self.features = nn.Sequential(
            resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool,
            resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4,
        )
        self.pool = nn.AdaptiveAvgPool2d(1)

        # Project to feature_dim, then classify
        resnet_out_dim = 512
        self.projector = nn.Sequential(
            nn.Linear(resnet_out_dim, feature_dim),
            nn.ReLU(),
            nn.Dropout(MLP_DROPOUT),
        )
        self.classifier = nn.Linear(feature_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, 13, H, W)"""
        h = self.encode(x)
        return self.classifier(h)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Return feature vector for fusion. (B, feature_dim)"""
        h = self.features(x)
        h = self.pool(h).flatten(1)  # (B, 512)
        return self.projector(h)     # (B, feature_dim)
