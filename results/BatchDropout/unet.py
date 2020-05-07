""" Full assembly of the parts to form the complete network """

import torch.nn.functional as F

from unet_parts import *


class UNet(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True):
        super(UNet, self).__init__()
        self.in_channels = in_channels
        self.bilinear = bilinear

        self.inc = nn.Sequential(
                    DoubleConv(in_channels, 64),
                    nn.Dropout2d(p=0.2),
        )
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 128)
        self.down3 = Down(128, 128)
        factor = 2 if bilinear else 1
        self.down4 = Down(128, 256 // factor)
        self.up1 = Up(256, 256 // factor, bilinear)
        self.up2 = Up(256, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, out_channels)

    def forward(self, x):
        # print('Batchsize {}'.format(x.size()[0]))
        inp = x
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        # print(x5.size())
        x = self.up1(x5, x4)
        # print(x.size())
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return inp[:, :3] + logits