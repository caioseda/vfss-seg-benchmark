'''
`segmentation_models_pytorch` architectures behind the repo's model contract.

E1 compares decoders on the same ImageNet-pretrained ResNet34 encoder, and smp provides U-Net,
U-Net++, FPN and DeepLabv3+ for exactly that encoder, with its decoder defaults serving as the
reference implementation. The adapter adds the two things the rest of the code expects: an
`n_classes` attribute (`LitWrapper` reads it) and the `[-1, 1]` -> ImageNet input conversion.
'''

import segmentation_models_pytorch as smp
from torch import nn

from ..modules import InputNormalization


class SMPModel(nn.Module):
    def __init__(
        self,
        arch: str,
        n_channels: int,
        n_classes: int,
        encoder_name: str = "resnet34",
        encoder_weights: str | None = "imagenet",
        input_normalization: str | None = None,
        **arch_params,
    ):
        '''
        Args:
            arch: smp architecture name, e.g. `Unet`, `UnetPlusPlus`, `FPN`, `DeepLabV3Plus`.
            encoder_weights: `imagenet`, or None for a random init (also what the tests use, to
                avoid a download).
            input_normalization: `imagenet` to standardise the dataset's `[-1, 1]` images as the
                pretrained encoder expects; None leaves them as they are.
            arch_params: forwarded to the smp constructor (decoder widths and the like).
        '''
        super().__init__()
        self.arch = arch
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.input_normalization = InputNormalization(input_normalization)
        self.model = smp.create_model(
            arch,
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=n_channels,
            classes=n_classes,
            **arch_params,
        )

    def forward(self, x):
        return self.model(self.input_normalization(x))
