"""UNet - Standard U-Net architecture for image segmentation."""
from __future__ import annotations

import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from framework.contracts import Batch, ModelOutput

from .parts.unet_parts import DoubleConv, Down, Up, OutConv


def dice_coeff(pred, gt, smooth=1e-5, activation="sigmoid"):
    """Calculate Dice coefficient."""
    if activation is None or activation == "none":
        activation_fn = lambda x: x
    elif activation == "sigmoid":
        activation_fn = nn.Sigmoid()
    elif activation == "softmax2d":
        activation_fn = nn.Softmax2d()
    else:
        raise NotImplementedError("Activation implemented for sigmoid and softmax2d")

    pred = activation_fn(pred)

    N = gt.size(0)
    pred_flat = pred.view(N, -1)
    gt_flat = gt.view(N, -1)

    intersection = (pred_flat * gt_flat).sum(1)
    unionset = pred_flat.sum(1) + gt_flat.sum(1)
    loss = (2 * intersection + smooth) / (unionset + smooth)

    return loss.sum() / N


class SoftDiceLoss(nn.Module):
    """Soft Dice loss for segmentation."""

    __name__ = "dice_loss"

    def __init__(self, activation="sigmoid"):
        super().__init__()
        self.activation = activation

    def forward(self, y_pr, y_gt):
        return 1 - dice_coeff(y_pr, y_gt, activation=self.activation)


class UNet(nn.Module):
    """Standard U-Net for image segmentation."""

    def __init__(self, n_channels: int = 3, n_classes: int = 17, bilinear: bool = True):
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 512)

        self.up1 = Up(1024, 256, bilinear)
        self.up2 = Up(512, 128, bilinear)
        self.up3 = Up(256, 64, bilinear)
        self.up4 = Up(128, 64, bilinear)
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
