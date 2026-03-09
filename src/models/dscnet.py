"""DSCNet - Dynamic Selection Convolution Network for segmentation."""
from __future__ import annotations

import torch
from torch import nn

from framework.contracts import Batch, ModelOutput

from .parts.dsconv import DSConv_pro


class EncoderConv(nn.Module):
    """Encoder convolution block with GroupNorm."""

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.gn = nn.GroupNorm(out_ch // 4, out_ch)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.gn(x)
        x = self.relu(x)
        return x


class DecoderConv(nn.Module):
    """Decoder convolution block with GroupNorm."""

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.gn = nn.GroupNorm(out_ch // 4, out_ch)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.gn(x)
        x = self.relu(x)
        return x


class DSCNet(nn.Module):
    """DSCNet with Dynamic Selection Convolution for medical image segmentation."""

    def __init__(
        self,
        n_channels: int = 3,
        n_classes: int = 17,
        kernel_size: int = 9,
        extend_scope: float = 1.0,
        if_offset: bool = True,
        device: str = "cuda",
        number: int = 32,
        dim: int = 1,
    ):
        super().__init__()
        self.kernel_size = kernel_size
        self.extend_scope = extend_scope
        self.if_offset = if_offset
        self.relu = nn.ReLU(inplace=True)
        self.number = number
        self.dim = dim

        self.conv00 = EncoderConv(n_channels, self.number)
        self.conv0x = DSConv_pro(
            n_channels,
            self.number,
            self.kernel_size,
            self.extend_scope,
            0,
            self.if_offset,
            device,
        )
        self.conv0y = DSConv_pro(
            n_channels,
            self.number,
            self.kernel_size,
            self.extend_scope,
            1,
            self.if_offset,
            device,
        )
        self.conv1 = EncoderConv(3 * self.number, self.number)

        self.conv20 = EncoderConv(self.number, 2 * self.number)
        self.conv2x = DSConv_pro(
            self.number,
            2 * self.number,
            self.kernel_size,
            self.extend_scope,
            0,
            self.if_offset,
            device,
        )
        self.conv2y = DSConv_pro(
            self.number,
            2 * self.number,
            self.kernel_size,
            self.extend_scope,
            1,
            self.if_offset,
            device,
        )
        self.conv3 = EncoderConv(6 * self.number, 2 * self.number)

        self.conv40 = EncoderConv(2 * self.number, 4 * self.number)
        self.conv4x = DSConv_pro(
            2 * self.number,
            4 * self.number,
            self.kernel_size,
            self.extend_scope,
            0,
            self.if_offset,
            device,
        )
        self.conv4y = DSConv_pro(
            2 * self.number,
            4 * self.number,
            self.kernel_size,
            self.extend_scope,
            1,
            self.if_offset,
            device,
        )
        self.conv5 = EncoderConv(12 * self.number, 4 * self.number)

        self.conv60 = EncoderConv(4 * self.number, 8 * self.number)
        self.conv6x = DSConv_pro(
            4 * self.number,
            8 * self.number,
            self.kernel_size,
            self.extend_scope,
            0,
            self.if_offset,
            device,
        )
        self.conv6y = DSConv_pro(
            4 * self.number,
            8 * self.number,
            self.kernel_size,
            self.extend_scope,
            1,
            self.if_offset,
            device,
        )
        self.conv7 = EncoderConv(24 * self.number, 8 * self.number)

        self.conv120 = EncoderConv(12 * self.number, 4 * self.number)
        self.conv12x = DSConv_pro(
            12 * self.number,
            4 * self.number,
            self.kernel_size,
            self.extend_scope,
            0,
            self.if_offset,
            device,
        )
        self.conv12y = DSConv_pro(
            12 * self.number,
            4 * self.number,
            self.kernel_size,
            self.extend_scope,
            1,
            self.if_offset,
            device,
        )
        self.conv13 = EncoderConv(12 * self.number, 4 * self.number)

        self.conv140 = DecoderConv(6 * self.number, 2 * self.number)
        self.conv14x = DSConv_pro(
            6 * self.number,
            2 * self.number,
            self.kernel_size,
            self.extend_scope,
            0,
            self.if_offset,
            device,
        )
        self.conv14y = DSConv_pro(
            6 * self.number,
            2 * self.number,
            self.kernel_size,
            self.extend_scope,
            1,
            self.if_offset,
            device,
        )
        self.conv15 = DecoderConv(6 * self.number, 2 * self.number)

        self.conv160 = DecoderConv(3 * self.number, self.number)
        self.conv16x = DSConv_pro(
            3 * self.number,
            self.number,
            self.kernel_size,
            self.extend_scope,
            0,
            self.if_offset,
            device,
        )
        self.conv16y = DSConv_pro(
            3 * self.number,
            self.number,
            self.kernel_size,
            self.extend_scope,
            1,
            self.if_offset,
            device,
        )
        self.conv17 = DecoderConv(3 * self.number, self.number)

        self.out_conv = nn.Conv2d(self.number, n_classes, 1)
        self.maxpooling = nn.MaxPool2d(2)
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, batch: Batch) -> ModelOutput:
        x = batch["image"]  # [B, C, H, W]

        # block0
        x_00_0 = self.conv00(x)
        x_0x_0 = self.conv0x(x)
        x_0y_0 = self.conv0y(x)
        x_0_1 = self.conv1(torch.cat([x_00_0, x_0x_0, x_0y_0], dim=1))

        # block1
        x = self.maxpooling(x_0_1)
        x_20_0 = self.conv20(x)
        x_2x_0 = self.conv2x(x)
        x_2y_0 = self.conv2y(x)
        x_1_1 = self.conv3(torch.cat([x_20_0, x_2x_0, x_2y_0], dim=1))

        # block2
        x = self.maxpooling(x_1_1)
        x_40_0 = self.conv40(x)
        x_4x_0 = self.conv4x(x)
        x_4y_0 = self.conv4y(x)
        x_2_1 = self.conv5(torch.cat([x_40_0, x_4x_0, x_4y_0], dim=1))

        # block3
        x = self.maxpooling(x_2_1)
        x_60_0 = self.conv60(x)
        x_6x_0 = self.conv6x(x)
        x_6y_0 = self.conv6y(x)
        x_3_1 = self.conv7(torch.cat([x_60_0, x_6x_0, x_6y_0], dim=1))

        # block4
        x = self.up(x_3_1)
        x_120_2 = self.conv120(torch.cat([x, x_2_1], dim=1))
        x_12x_2 = self.conv12x(torch.cat([x, x_2_1], dim=1))
        x_12y_2 = self.conv12y(torch.cat([x, x_2_1], dim=1))
        x_2_3 = self.conv13(torch.cat([x_120_2, x_12x_2, x_12y_2], dim=1))

        # block5
        x = self.up(x_2_3)
        x_140_2 = self.conv140(torch.cat([x, x_1_1], dim=1))
        x_14x_2 = self.conv14x(torch.cat([x, x_1_1], dim=1))
        x_14y_2 = self.conv14y(torch.cat([x, x_1_1], dim=1))
        x_1_3 = self.conv15(torch.cat([x_140_2, x_14x_2, x_14y_2], dim=1))

        # block6
        x = self.up(x_1_3)
        x_160_2 = self.conv160(torch.cat([x, x_0_1], dim=1))
        x_16x_2 = self.conv16x(torch.cat([x, x_0_1], dim=1))
        x_16y_2 = self.conv16y(torch.cat([x, x_0_1], dim=1))
        x_0_3 = self.conv17(torch.cat([x_160_2, x_16x_2, x_16y_2], dim=1))

        out = self.out_conv(x_0_3)
        out = self.sigmoid(out)

        return out
