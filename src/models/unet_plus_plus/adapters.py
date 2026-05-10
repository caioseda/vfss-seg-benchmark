from torch import nn

from .unetplusplus import Generic_UNetPlusPlus


class UNetPlusPlus(nn.Module):
    def __init__(
        self,
        n_channels: int,
        n_classes: int,
        base_num_features: int = 32,
        num_pool: int = 5,
        num_conv_per_stage: int = 2,
        feat_map_mul_on_downscale: int = 2,
        deep_supervision: bool = False,
    ):
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        self.model = Generic_UNetPlusPlus(
            input_channels=n_channels,
            base_num_features=base_num_features,
            num_classes=n_classes,
            num_pool=num_pool,
            num_conv_per_stage=num_conv_per_stage,
            feat_map_mul_on_downscale=feat_map_mul_on_downscale,
            deep_supervision=deep_supervision,
            final_nonlin=lambda x: x,
        )

    def forward(self, x):
        return self.model(x)
