from math import log2

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.networks.base_nets import (
    BaseGenerator,
    PixelShuffleUpsample,
    is_power_of_two,
    leaky_relu,
)


# Frequency Enhancement (FE) Operation
class FE(nn.Module):
    def __init__(self, in_channels, channels):
        super(FE, self).__init__()
        self.pool = nn.AvgPool2d(kernel_size=4, stride=4)

        self.k2 = nn.Conv2d(
            in_channels, channels, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.k3 = nn.Conv2d(
            in_channels, channels, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.k4 = nn.Conv2d(
            in_channels, channels, kernel_size=3, stride=1, padding=1, bias=False
        )

    def forward(self, x: torch.Tensor):
        h1 = F.interpolate(self.pool(x), (x.size(-2), x.size(-1)), mode="nearest")
        h2 = x - h1
        F2 = torch.sigmoid(torch.add(self.k2(h2), x))
        out = torch.mul(self.k3(x), F2)
        out = self.k4(out)

        return out


# Frequency-based Enhancement Block (FEB)
class FEB(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(FEB, self).__init__()
        channels = out_channels // 2
        self.path_1 = nn.Conv2d(in_channels, channels, kernel_size=1, bias=False)
        self.path_2 = nn.Conv2d(in_channels, channels, kernel_size=1, bias=False)
        self.act = leaky_relu()
        self.k1 = nn.Conv2d(
            channels, channels, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.HConv = FE(32, 32)
        self.conv = nn.Conv2d(channels * 2, out_channels, kernel_size=1, bias=False)

    def forward(self, x: torch.Tensor):
        # Low-Frequency Path
        low = self.path_1(x)
        low = self.act(low)
        low = self.k1(low)
        low = self.act(low)

        # High-Frequency Path
        high = self.path_2(x)
        high = self.act(high)
        high = self.HConv(high)
        high = self.act(high)

        output = self.conv(torch.cat([low, high], dim=1))
        output = output + x

        return output, low, high


class FENet(BaseGenerator):
    def __init__(
        self,
        in_channels=3,
        out_channels=3,
        *,
        feat_dim=64,
        n_feb_block=12,
        scale=4,
    ):
        super(FENet, self).__init__()

        self.scale = scale

        self.init_conv = nn.Conv2d(in_channels, feat_dim, 7, padding=3)

        self.febs = nn.Sequential(
            *[FEB(feat_dim, feat_dim) for _ in range(n_feb_block)]
        )
        self.reduction = nn.Conv2d(feat_dim * n_feb_block, feat_dim, 1)

        self.ups = nn.ModuleList([])

        assert is_power_of_two(scale), "scale should be power of 2"

        for _ in range(int(log2(scale))):
            self.ups.append(
                PixelShuffleUpsample(feat_dim),
            )

        self.exit = nn.Conv2d(feat_dim, out_channels, 3, 1, 1)

        self.high_f, self.low_f = [], []

    def forward(self, x):
        self.low_f.clear()
        self.high_f.clear()

        out = self.init_conv(x)

        c0 = out

        out_blocks = []

        for feb in self.febs:
            out, low_f, high_f = feb(out)
            self.low_f.append(low_f)
            self.high_f.append(high_f)
            out_blocks.append(out)

        output = self.reduction(torch.cat(out_blocks, 1))

        output = output + c0

        for upsample in self.ups:
            output = upsample(output)

        output = self.exit(output)

        skip = F.interpolate(
            x,
            scale_factor=self.scale,
            mode="bicubic",
            align_corners=False,
        )

        output = skip + output

        output = F.tanh(output)

        return output
