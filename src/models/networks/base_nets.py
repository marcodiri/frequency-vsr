from math import log2

import lightning as L
import torch
import torch.nn as nn
from einops import repeat


class BaseGenerator(L.LightningModule):
    pass


class BaseDiscriminator(L.LightningModule):
    pass


# helpers
def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d


def is_power_of_two(n):
    return log2(n).is_integer()


# blocks
def leaky_relu(neg_slope=0.2):
    return nn.LeakyReLU(neg_slope)


class PixelShuffleUpsample(nn.Module):
    def __init__(self, dim, dim_out=None):
        super().__init__()
        dim_out = default(dim_out, dim)

        conv = nn.Conv2d(dim, dim_out * 4, 1)
        self.init_conv_(conv)

        self.net = nn.Sequential(conv, nn.SiLU(), nn.PixelShuffle(2))

    def init_conv_(self, conv):
        o, *rest_shape = conv.weight.shape
        conv_weight = torch.empty(o // 4, *rest_shape)
        nn.init.kaiming_uniform_(conv_weight)
        conv_weight = repeat(conv_weight, "o ... -> (o 4) ...")

        conv.weight.data.copy_(conv_weight)
        nn.init.zeros_(conv.bias.data)

    def forward(self, x):
        return self.net(x)
