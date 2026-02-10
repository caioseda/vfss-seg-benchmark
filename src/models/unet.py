from torch import nn
from .modules import *

class UNet(nn.Module):
    def __init__(self, n_channels: int, n_classes: int, bilinear: bool = True):
        super().__init__()

        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        factor = 2 if bilinear else 1

        self.double_conv = DoubleConv(n_channels, 64)
        self.down1 = DownBlock(64, 128)
        self.down2 = DownBlock(128, 256)
        self.down3 = DownBlock(256, 512)
        self.down4 = DownBlock(512, 1024 // factor) # 512 -> 512
        self.up1 = UpBlock(1024, 512 // factor, bilinear) # 1024 -> 256
        self.up2 = UpBlock(512, 256 // factor, bilinear) # 256 -> 128
        self.up3 = UpBlock(256, 128 // factor, bilinear) # 128 -> 64
        self.up4 = UpBlock(128, 64, bilinear)
        self.out_conv = nn.Conv2d(64, n_classes, kernel_size=1)

    def forward(self, x):
        x_double = self.double_conv(x)
        x_down1 = self.down1(x_double)
        x_down2 = self.down2(x_down1)
        x_down3 = self.down3(x_down2)
        x_down4 = self.down4(x_down3)
        x_up1 = self.up1(x_down4, x_down3)
        x_up2 = self.up2(x_up1, x_down2)
        x_up3 = self.up3(x_up2, x_down1)
        x_up4 = self.up4(x_up3, x_double)
        x_out = self.out_conv(x_up4)
        return x_out
