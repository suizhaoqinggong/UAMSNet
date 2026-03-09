"""UAMSNet - Main segmentation model with MBA (Multi-scale Boundary Attention)."""
from __future__ import annotations

import torch.nn as nn
from torch import Tensor

from framework.contracts import Batch, ModelOutput

from .parts.unet_parts import ConvWithMBA, EcMBA, BnDCN_Context, UpMBA, OutConv


class UAMSNet(nn.Module):
    """UAMSNet with Multi-scale Boundary Attention for medical image segmentation."""

    def __init__(self, n_channels: int = 3, n_classes: int = 17, bilinear: bool = True):
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = ConvWithMBA(n_channels, 64)
        self.down1 = EcMBA(64, 128)
        self.down2 = EcMBA(128, 256)
        self.down3 = EcMBA(256, 512)
        self.down4 = BnDCN_Context(512, 512)
        self.up1 = UpMBA(1024, 256, bilinear)
        self.up2 = UpMBA(512, 128, bilinear)
        self.up3 = UpMBA(256, 64, bilinear)
        self.up4 = UpMBA(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, batch: Batch) -> ModelOutput:
        x = batch["image"]  # [B, C, H, W]
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits
