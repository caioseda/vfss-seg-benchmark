import torch
from torch import nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, mid_channels=None):
        super().__init__()

        if not mid_channels:
            mid_channels = out_channels

        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)


class DownBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.down = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels),
        )

    def forward(self, x):
        return self.down(x)


class UpBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, bilinear=True):
        super().__init__()

        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            self.double_conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(
                in_channels, in_channels // 2, kernel_size=2, stride=2
            )
            self.double_conv = DoubleConv(in_channels, out_channels)

    def forward(self, x_bn, x_skip):
        x_bn = self.up(x_bn)

        x_diff = x_skip.size()[2] - x_bn.size()[2]
        y_diff = x_skip.size()[3] - x_bn.size()[3]

        padding_left = x_diff // 2
        padding_right = x_diff - padding_left
        padding_top = y_diff // 2
        padding_bottom = y_diff - padding_top

        x_skip = F.pad(
            x_skip, [padding_left, padding_right, padding_top, padding_bottom]
        )
        out = torch.cat([x_skip, x_bn], dim=1)
        out = self.double_conv(out)
        return out


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


INPUT_NORMALIZATIONS = (None, "imagenet")
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class InputNormalization(nn.Module):
    '''
    Map the dataset's `[-1, 1]` images to what a pretrained encoder was trained on.

    `VFSSFrameDatasetBase._preprocess_image` min-max scales each frame to `[-1, 1]`, and that stays
    the dataset contract for every model. An ImageNet-pretrained encoder instead expects `[0, 1]`
    standardised by the ImageNet channel statistics, so the conversion lives in the model adapter,
    on the model's side of the contract, and only models that ask for it get it.

    `mode=None` is the identity. The buffers are non-persistent, so wrapping a model in this never
    changes its `state_dict` keys and older checkpoints keep loading.
    '''

    def __init__(self, mode=None):
        super().__init__()
        if mode not in INPUT_NORMALIZATIONS:
            raise ValueError(f"input_normalization must be one of {INPUT_NORMALIZATIONS}, got {mode!r}")
        self.mode = mode
        imagenet_mean = torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
        imagenet_std = torch.tensor(IMAGENET_STD).view(1, 3, 1, 1)
        
        if mode == "imagenet":
            self.register_buffer("mean", imagenet_mean, persistent=False)
            self.register_buffer("std", imagenet_std, persistent=False)

    def forward(self, x):
        if self.mode is None:
            return x

        x_normalized = (x + 1.0) / 2.0  # [-1, 1] -> [0, 1]
        x_standardized = (x_normalized - self.mean) / self.std  # [0, 1] -> ImageNet standardization
        return x_standardized
        
