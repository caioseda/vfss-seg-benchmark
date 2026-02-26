from copy import deepcopy

from omegaconf import DictConfig, OmegaConf
import torch
from torch import nn

from .transunet import VisionTransformer

class TransUNet(nn.Module):
    def __init__(
        self,
        n_channels: int,
        n_classes: int,
        image_size: int = 256,
        patch_size: int | None = None,
        vis: bool = False,
        vit_config: dict | DictConfig | None = None,
    ):
        super().__init__()
        if vit_config is None:
            raise ValueError("`vit_config` must be provided from YAML for TransUNet.")

        config = OmegaConf.create(deepcopy(vit_config))
        config.n_classes = n_classes
        if patch_size is not None:
            config.patch_size = patch_size
            config.patches.size = [patch_size, patch_size]

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
        if config.get("pretrained_path"):
            try:
                checkpoint = torch.load(config.pretrained_path, map_location="cpu", weights_only=True)
            except TypeError:
                checkpoint = torch.load(config.pretrained_path, map_location="cpu")
            if isinstance(checkpoint, dict) and "state_dict" in checkpoint and "vit.embeddings.patch_embeddings.projection.weight" not in checkpoint:
                checkpoint = checkpoint["state_dict"]
            self.model.load_from(weights=checkpoint)

    def forward(self, x):
        return self.model(x)
