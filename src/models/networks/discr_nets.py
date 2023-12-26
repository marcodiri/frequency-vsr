import torch
import torch.nn as nn

from models.networks.base_nets import BaseDiscriminator, leaky_relu


class DiscriminatorBlocks(nn.Module):
    def __init__(self):
        super(DiscriminatorBlocks, self).__init__()

        self.block1 = nn.Sequential(  # /2
            nn.Conv2d(64, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.block2 = nn.Sequential(  # /4
            nn.Conv2d(64, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.block3 = nn.Sequential(  # /8
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.block4 = nn.Sequential(  # /16
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x):
        out = self.block1(x)
        out = self.block2(out)
        out = self.block3(out)
        out = self.block4(out)

        return out


class SpatialDiscriminator(BaseDiscriminator):
    """Spatial discriminator"""

    def __init__(self, in_nc=3, spatial_size=128, use_cond=False):
        super(SpatialDiscriminator, self).__init__()

        # basic settings
        self.use_cond = use_cond  # whether to use conditional input
        mult = 2 if self.use_cond else 1

        # input conv
        self.conv_in = nn.Sequential(
            nn.Conv2d(in_nc * mult, 64, 3, 1, 1, bias=True), leaky_relu()
        )

        # discriminator block
        self.discriminator_block = DiscriminatorBlocks()  # /16

        # classifier
        self.dense = nn.Linear(256 * spatial_size // 16 * spatial_size // 16, 1)

    def forward_single(self, x):
        out = self.conv_in(x)
        out = self.discriminator_block(out)

        out = out.view(out.size(0), -1)
        out = self.dense(out)

        return out

    def forward(self, data, **kwargs):
        # ------------ build up inputs for D ------------ #
        if self.use_cond:
            bi_data = kwargs["bi_data"]
            input_data = torch.cat([bi_data, data], dim=1)
        else:
            input_data = data

        # ------------ classify ------------ #
        pred = self.forward_single(input_data)

        return pred
