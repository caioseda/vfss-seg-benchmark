from copy import deepcopy

import torch
from omegaconf import DictConfig, OmegaConf
from torch import nn

from .modules import SwinTransformerSys


class SwinUNet(nn.Module):
    def __init__(
        self,
        n_channels: int,
        n_classes: int,
        image_size: int = 256,
        pretrain_ckpt: str | None = None,
        drop_rate: float = 0.0,
        drop_path_rate: float = 0.1,
        use_checkpoint: bool = False,
        swin: dict | DictConfig | None = None,
    ):
        super().__init__()
        if swin is None:
            raise ValueError("`swin` configuration must be provided from YAML for SwinUNet.")

        swin_cfg = OmegaConf.to_container(deepcopy(swin), resolve=True)
        if swin_cfg is None:
            raise ValueError("`swin` configuration is empty.")

        self.n_channels = n_channels
        self.n_classes = n_classes

        self.model = SwinTransformerSys(
            img_size=image_size,
            patch_size=swin_cfg.get("patch_size", 4),
            in_chans=swin_cfg.get("in_chans", n_channels),
            num_classes=n_classes,
            embed_dim=swin_cfg.get("embed_dim", 96),
            depths=swin_cfg.get("depths", [2, 2, 2, 2]),
            depths_decoder=swin_cfg.get("decoder_depths", [1, 2, 2, 2]),
            num_heads=swin_cfg.get("num_heads", [3, 6, 12, 24]),
            window_size=swin_cfg.get("window_size", 7),
            mlp_ratio=swin_cfg.get("mlp_ratio", 4.0),
            qkv_bias=swin_cfg.get("qkv_bias", True),
            qk_scale=swin_cfg.get("qk_scale", None),
            drop_rate=drop_rate,
            drop_path_rate=drop_path_rate,
            ape=swin_cfg.get("ape", False),
            patch_norm=swin_cfg.get("patch_norm", True),
            use_checkpoint=use_checkpoint,
            final_upsample=swin_cfg.get("final_upsample", "expand_first"),
        )

        if pretrain_ckpt:
            self._load_pretrained(pretrain_ckpt)

    def _load_pretrained(self, checkpoint_path: str) -> None:
        try:
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        except TypeError:
            checkpoint = torch.load(checkpoint_path, map_location="cpu")

        if isinstance(checkpoint, dict):
            if "model" in checkpoint and isinstance(checkpoint["model"], dict):
                pretrained_state = checkpoint["model"]
            elif "state_dict" in checkpoint and isinstance(checkpoint["state_dict"], dict):
                pretrained_state = checkpoint["state_dict"]
            else:
                pretrained_state = checkpoint
        else:
            pretrained_state = checkpoint

        cleaned_state = {}
        for key, value in pretrained_state.items():
            new_key = key
            for prefix in ("module.", "model.", "swin_unet."):
                if new_key.startswith(prefix):
                    new_key = new_key[len(prefix):]
            cleaned_state[new_key] = value

        full_state = dict(cleaned_state)
        num_layers = len(self.model.layers)
        for key, value in cleaned_state.items():
            if key.startswith("layers."):
                parts = key.split(".")
                if len(parts) > 1 and parts[1].isdigit():
                    src_idx = int(parts[1])
                    dst_idx = num_layers - 1 - src_idx
                    mirrored_key = key.replace(f"layers.{src_idx}", f"layers_up.{dst_idx}", 1)
                    full_state[mirrored_key] = value

        model_state = self.model.state_dict()
        compatible_state = {
            key: value
            for key, value in full_state.items()
            if key in model_state and getattr(value, "shape", None) == model_state[key].shape
        }
        self.model.load_state_dict(compatible_state, strict=False)
        print(f"Loaded pretrained weights from {checkpoint_path} with {len(compatible_state)} compatible parameters.")

    def forward(self, x):
        if x.size(1) == 1 and self.n_channels == 3:
            x = x.repeat(1, 3, 1, 1)
        return self.model(x)
