from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class BaselineUNet(nn.Module):
    def __init__(self, base: int = 12) -> None:
        super().__init__()
        self.enc1 = ConvBlock(3, base)
        self.enc2 = ConvBlock(base, base * 2)
        self.enc3 = ConvBlock(base * 2, base * 4)
        self.bottleneck = ConvBlock(base * 4, base * 8)
        self.pool = nn.MaxPool2d(2)
        self.dec3 = ConvBlock(base * 12, base * 4)
        self.dec2 = ConvBlock(base * 6, base * 2)
        self.dec1 = ConvBlock(base * 3, base)
        self.head = nn.Conv2d(base, 1, 1)

    def _up(self, x, skip):
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return torch.cat([x, skip], dim=1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        b = self.bottleneck(self.pool(e3))
        d3 = self.dec3(self._up(b, e3))
        d2 = self.dec2(self._up(d3, e2))
        d1 = self.dec1(self._up(d2, e1))
        return {"mask": self.head(d1)}


class DSResidualBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.depthwise = nn.Conv2d(in_ch, in_ch, 3, padding=1, groups=in_ch, bias=False)
        self.pointwise = nn.Conv2d(in_ch, out_ch, 1, bias=False)
        self.norm1 = nn.BatchNorm2d(in_ch)
        self.norm2 = nn.BatchNorm2d(out_ch)
        self.act = nn.SiLU(inplace=True)
        self.proj = nn.Identity() if in_ch == out_ch else nn.Conv2d(in_ch, out_ch, 1, bias=False)
        self.refine = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, 3, padding=1, groups=out_ch, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
        )

    def forward(self, x):
        residual = self.proj(x)
        y = self.act(self.norm1(self.depthwise(x)))
        y = self.act(self.norm2(self.pointwise(y)))
        return self.act(self.refine(y) + residual)


class ResidualConvBlock(nn.Module):
    """Two full convolutions with an identity/projection residual path."""

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
        )
        self.proj = nn.Identity() if in_ch == out_ch else nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 1, bias=False), nn.BatchNorm2d(out_ch)
        )
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        return self.act(self.main(x) + self.proj(x))


class CBAM(nn.Module):
    def __init__(self, channels: int, reduction: int = 8) -> None:
        super().__init__()
        hidden = max(channels // reduction, 4)
        self.mlp = nn.Sequential(nn.Conv2d(channels, hidden, 1), nn.ReLU(), nn.Conv2d(hidden, channels, 1))
        self.spatial = nn.Conv2d(2, 1, 7, padding=3)

    def forward(self, x):
        channel = torch.sigmoid(self.mlp(F.adaptive_avg_pool2d(x, 1)) + self.mlp(F.adaptive_max_pool2d(x, 1)))
        x = x * channel
        spatial = torch.sigmoid(self.spatial(torch.cat([x.mean(1, keepdim=True), x.amax(1, keepdim=True)], 1)))
        return x * spatial


class EdgeAttentionUNet(nn.Module):
    """Lightweight residual U-Net with attention-gated skips and an auxiliary edge head."""

    def __init__(self, base: int = 12) -> None:
        super().__init__()
        self.enc1 = ResidualConvBlock(3, base)
        self.enc2 = ResidualConvBlock(base, base * 2)
        self.enc3 = ResidualConvBlock(base * 2, base * 4)
        self.bottleneck = ResidualConvBlock(base * 4, base * 8)
        self.pool = nn.MaxPool2d(2)
        self.att3, self.att2, self.att1 = CBAM(base * 4), CBAM(base * 2), CBAM(base)
        self.dec3 = ResidualConvBlock(base * 12, base * 4)
        self.dec2 = ResidualConvBlock(base * 6, base * 2)
        self.dec1 = ResidualConvBlock(base * 3, base)
        self.mask_head = nn.Conv2d(base, 1, 1)
        self.edge_head = nn.Sequential(nn.Conv2d(base, base // 2, 3, padding=1), nn.SiLU(), nn.Conv2d(base // 2, 1, 1))

    @staticmethod
    def _up(x, skip):
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return torch.cat([x, skip], dim=1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        b = self.bottleneck(self.pool(e3))
        d3 = self.dec3(self._up(b, self.att3(e3)))
        d2 = self.dec2(self._up(d3, self.att2(e2)))
        d1 = self.dec1(self._up(d2, self.att1(e1)))
        return {"mask": self.mask_head(d1), "edge": self.edge_head(d1)}


def build_model(name: str, base: int = 12) -> nn.Module:
    if name == "baseline":
        return BaselineUNet(base)
    if name == "improved":
        return EdgeAttentionUNet(base)
    raise ValueError(f"Unknown model: {name}")

