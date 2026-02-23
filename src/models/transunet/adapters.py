from copy import deepcopy

from omegaconf import DictConfig, OmegaConf
from torch import nn

from .official_transunet import VisionTransformer

class TransUNet(nn.Module):
    def __init__(
        self,
        n_channels: int,
        n_classes: int,
        image_size: int = 256,
        vis: bool = False,
        vit_config: dict | DictConfig | None = None,
    ):
        super().__init__()
        if vit_config is None:
            raise ValueError("`vit_config` must be provided from YAML for TransUNet.")

        config = OmegaConf.create(deepcopy(vit_config))
        config.n_classes = n_classes

        if config.get("n_skip") is None:
            config.n_skip = 0
        if config.get("skip_channels") is None:
            config.skip_channels = [0, 0, 0, 0]

        self.n_channels = n_channels
        self.n_classes = n_classes
        self.model = VisionTransformer(
            config=config,
            img_size=image_size,
            num_classes=n_classes,
            vis=vis,
        )

    def forward(self, x):
        return self.model(x)
