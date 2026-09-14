'''
Adapted from the official SSL4MIS implementation:
    https://github.com/HiLab-git/SSL4MIS  --  code/networks/unet.py
    commit 06df6047a59aba9988ced8331998b5957ecb356b
    License: MIT
(SSL4MIS in turn credits https://github.com/HiLab-git/PyMIC.)

This is the backbone **every** number in the 2-D ACDC semi-supervised literature is measured on,
DiffRect's included, and it is not the same network as `src/models/unet.py`:

| | this file | `src/models/unet.py` |
|---|---|---|
| stage widths | 16, 32, 64, 128, 256 | 64, 128, 256, 512, 512 |
| parameters (1 in, 4 classes) | **1.81 M** | **17.26 M** |
| dropout | 0.05 / 0.1 / 0.2 / 0.3 / 0.5 | none |
| activation | LeakyReLU | ReLU |
| upsampling | bilinear + 1x1 conv | bilinear + 1x1 conv |

A 9.5x wider network with no dropout is a different experiment at a 32-slice label budget, so
comparing our number against a published one while silently swapping the backbone would not be a
reproduction. It is vendored here so the ACDC configs can run the reference architecture and ours
side by side, and the gap between them is measured rather than assumed.

`src/models/unet.py` stays the backbone for X1/X1B: there the question is how the *methods* compare
on VFSS under a fixed architecture, not how a published number reproduces.

Deviations from the original, all mechanical:
  - constructor takes `n_channels` / `n_classes` and exposes them as attributes, which is the
    contract `LitWrapper` and `instantiate_from_config` expect (upstream: `in_chns` / `class_num`);
  - `forward` returns the logits directly (upstream returns a list under `out_multi=True` and then
    indexes `[-1]`), and the unused `save_feature_iter` / `iter_num` / `strong` arguments are gone;
  - the deep-supervision, URPC and CCT variants are not vendored -- nothing here uses them.
The layer arithmetic (widths, dropout schedule, kernel sizes, LeakyReLU, transposed-conv
upsampling) is unchanged.
'''

import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    '''Two convolution layers with batch norm and leaky relu.'''

    def __init__(self, in_channels, out_channels, dropout_p):
        super().__init__()
        self.conv_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(),
            nn.Dropout(dropout_p),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(),
        )

    def forward(self, x):
        return self.conv_conv(x)


class DownBlock(nn.Module):
    '''Downsampling followed by ConvBlock.'''

    def __init__(self, in_channels, out_channels, dropout_p):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            ConvBlock(in_channels, out_channels, dropout_p),
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class UpBlock(nn.Module):
    '''Upsampling followed by ConvBlock.'''

    def __init__(self, in_channels1, in_channels2, out_channels, dropout_p, bilinear=True):
        super().__init__()
        self.bilinear = bilinear
        if bilinear:
            self.conv1x1 = nn.Conv2d(in_channels1, in_channels2, kernel_size=1)
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_channels1, in_channels2, kernel_size=2, stride=2)
        self.conv = ConvBlock(in_channels2 * 2, out_channels, dropout_p)

    def forward(self, x1, x2):
        if self.bilinear:
            x1 = self.conv1x1(x1)
        x1 = self.up(x1)
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class Encoder(nn.Module):
    def __init__(self, params):
        super().__init__()
        self.in_chns = params["in_chns"]
        self.ft_chns = params["feature_chns"]
        self.dropout = params["dropout"]
        assert len(self.ft_chns) == 5

        self.in_conv = ConvBlock(self.in_chns, self.ft_chns[0], self.dropout[0])
        self.down1 = DownBlock(self.ft_chns[0], self.ft_chns[1], self.dropout[1])
        self.down2 = DownBlock(self.ft_chns[1], self.ft_chns[2], self.dropout[2])
        self.down3 = DownBlock(self.ft_chns[2], self.ft_chns[3], self.dropout[3])
        self.down4 = DownBlock(self.ft_chns[3], self.ft_chns[4], self.dropout[4])

    def forward(self, x):
        x0 = self.in_conv(x)
        x1 = self.down1(x0)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x4 = self.down4(x3)
        return [x0, x1, x2, x3, x4]


class Decoder(nn.Module):
    def __init__(self, params):
        super().__init__()
        self.ft_chns = params["feature_chns"]
        self.n_class = params["class_num"]
        self.bilinear = params["bilinear"]
        assert len(self.ft_chns) == 5

        # Upstream does NOT forward `params['bilinear']` to the UpBlocks, so they all take
        # `UpBlock`'s default of `bilinear=True` and `params['bilinear']` is dead code. Transcribed
        # as-is: reproducing the network the published numbers were measured on matters more than
        # honouring a flag upstream ignores. Passing `bilinear=False` here instead would swap in
        # transposed convolutions and take the model from 1.81 M to 1.94 M parameters.
        self.up1 = UpBlock(self.ft_chns[4], self.ft_chns[3], self.ft_chns[3], dropout_p=0.0)
        self.up2 = UpBlock(self.ft_chns[3], self.ft_chns[2], self.ft_chns[2], dropout_p=0.0)
        self.up3 = UpBlock(self.ft_chns[2], self.ft_chns[1], self.ft_chns[1], dropout_p=0.0)
        self.up4 = UpBlock(self.ft_chns[1], self.ft_chns[0], self.ft_chns[0], dropout_p=0.0)
        self.out_conv = nn.Conv2d(self.ft_chns[0], self.n_class, kernel_size=3, padding=1)

    def forward(self, feature):
        x0, x1, x2, x3, x4 = feature
        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        x = self.up4(x, x0)
        return self.out_conv(x)


class SSL4MISUNet(nn.Module):
    '''
    The SSL4MIS / DiffRect U-Net. 1.81 M parameters at `n_channels=1, n_classes=4`.

    There is deliberately no `bilinear` argument: upstream accepts one and then never forwards it
    to the decoder's up-blocks, so it changes nothing. Exposing it here would advertise a knob that
    does not exist.

    Args:
        n_channels: input channels (1 for ACDC MRI).
        n_classes: including background.
    '''

    def __init__(self, n_channels: int, n_classes: int):
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        params = {
            "in_chns": n_channels,
            "feature_chns": [16, 32, 64, 128, 256],
            "dropout": [0.05, 0.1, 0.2, 0.3, 0.5],
            "class_num": n_classes,
            "bilinear": False,   # dead upstream; see the note in `Decoder.__init__`
            "acti_func": "relu",
        }
        self.encoder = Encoder(params)
        self.decoder = Decoder(params)

    def forward(self, x):
        return self.decoder(self.encoder(x))
