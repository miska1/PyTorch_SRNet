"""PyTorch implementation of the original 12-layer SRNet architecture."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _conv3x3(in_channels: int, out_channels: int) -> nn.Conv2d:
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=True)


class Type1(nn.Module):
    """3x3 convolution, batch normalization, and ReLU."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv = _conv3x3(in_channels, out_channels)
        # PyTorch momentum=0.1 corresponds to an EMA decay of 0.9.
        self.bn = nn.BatchNorm2d(out_channels, momentum=0.1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return F.relu(self.bn(self.conv(inputs)), inplace=True)


class Type2(nn.Module):
    """Unpooled SRNet residual layer."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.first = Type1(channels, channels)
        self.conv = _conv3x3(channels, channels)
        self.bn = nn.BatchNorm2d(channels, momentum=0.1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return inputs + self.bn(self.conv(self.first(inputs)))


class Type3(nn.Module):
    """Downsampling SRNet residual layer."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.first = Type1(in_channels, out_channels)
        self.conv = _conv3x3(out_channels, out_channels)
        self.main_bn = nn.BatchNorm2d(out_channels, momentum=0.1)
        self.shortcut_conv = nn.Conv2d(
            in_channels, out_channels, kernel_size=1, stride=2, bias=True
        )
        self.shortcut_bn = nn.BatchNorm2d(out_channels, momentum=0.1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        main = self.main_bn(self.conv(self.first(inputs)))
        main = F.avg_pool2d(main, kernel_size=3, stride=2, padding=1)
        shortcut = self.shortcut_bn(self.shortcut_conv(inputs))
        return main + shortcut


class Type4(nn.Module):
    """Final SRNet layer with global average pooling."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.first = Type1(in_channels, out_channels)
        self.conv = _conv3x3(out_channels, out_channels)
        self.bn = nn.BatchNorm2d(out_channels, momentum=0.1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        features = self.bn(self.conv(self.first(inputs)))
        return F.adaptive_avg_pool2d(features, output_size=1).flatten(1)


class SRNet(nn.Module):
    """Classify one-channel images as cover (0) or stego (1)."""

    def __init__(self, num_classes: int = 2) -> None:
        super().__init__()
        self.layer1 = Type1(1, 64)
        self.layer2 = Type1(64, 16)
        self.unpooled = nn.Sequential(*(Type2(16) for _ in range(5)))
        self.pooled = nn.Sequential(
            Type3(16, 16),
            Type3(16, 64),
            Type3(64, 128),
            Type3(128, 256),
        )
        self.layer12 = Type4(256, 512)
        self.classifier = nn.Linear(512, num_classes, bias=False)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        #initialization of weights
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.2)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.01)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        features = self.layer1(inputs)
        features = self.layer2(features)
        features = self.unpooled(features)
        features = self.pooled(features)
        features = self.layer12(features)
        return self.classifier(features)
