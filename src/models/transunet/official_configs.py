from omegaconf import DictConfig, OmegaConf


def _cfg(data) -> DictConfig:
    return OmegaConf.create(data)


def get_b16_config() -> DictConfig:
    """Returns the ViT-B/16 configuration."""
    return _cfg(
        {
            "patches": {"size": (16, 16)},
            "hidden_size": 768,
            "transformer": {
                "mlp_dim": 3072,
                "num_heads": 12,
                "num_layers": 12,
                "attention_dropout_rate": 0.0,
                "dropout_rate": 0.1,
            },
            "classifier": "seg",
            "representation_size": None,
            "resnet_pretrained_path": None,
            "pretrained_path": "../model/vit_checkpoint/imagenet21k/ViT-B_16.npz",
            "patch_size": 16,
            "decoder_channels": (256, 128, 64, 16),
            "n_classes": 2,
            "activation": "softmax",
        }
    )


def get_testing() -> DictConfig:
    """Returns a minimal configuration for testing."""
    return _cfg(
        {
            "patches": {"size": (16, 16)},
            "hidden_size": 1,
            "transformer": {
                "mlp_dim": 1,
                "num_heads": 1,
                "num_layers": 1,
                "attention_dropout_rate": 0.0,
                "dropout_rate": 0.1,
            },
            "classifier": "token",
            "representation_size": None,
        }
    )


def get_r50_b16_config() -> DictConfig:
    """Returns the Resnet50 + ViT-B/16 configuration."""
    config = get_b16_config()
    config.patches.grid = (16, 16)
    config.resnet = _cfg(
        {
            "num_layers": (3, 4, 9),
            "width_factor": 1,
        }
    )

    config.classifier = "seg"
    config.pretrained_path = "../model/vit_checkpoint/imagenet21k/R50+ViT-B_16.npz"
    config.decoder_channels = (256, 128, 64, 16)
    config.skip_channels = [512, 256, 64, 16]
    config.n_classes = 2
    config.n_skip = 3
    config.activation = "softmax"

    return config


def get_b32_config() -> DictConfig:
    """Returns the ViT-B/32 configuration."""
    config = get_b16_config()
    config.patches.size = (32, 32)
    config.pretrained_path = "../model/vit_checkpoint/imagenet21k/ViT-B_32.npz"
    return config


def get_l16_config() -> DictConfig:
    """Returns the ViT-L/16 configuration."""
    return _cfg(
        {
            "patches": {"size": (16, 16)},
            "hidden_size": 1024,
            "transformer": {
                "mlp_dim": 4096,
                "num_heads": 16,
                "num_layers": 24,
                "attention_dropout_rate": 0.0,
                "dropout_rate": 0.1,
            },
            "representation_size": None,
            "classifier": "seg",
            "resnet_pretrained_path": None,
            "pretrained_path": "../model/vit_checkpoint/imagenet21k/ViT-L_16.npz",
            "decoder_channels": (256, 128, 64, 16),
            "n_classes": 2,
            "activation": "softmax",
        }
    )


def get_r50_l16_config() -> DictConfig:
    """Returns the Resnet50 + ViT-L/16 configuration. customized """
    config = get_l16_config()
    config.patches.grid = (16, 16)
    config.resnet = _cfg(
        {
            "num_layers": (3, 4, 9),
            "width_factor": 1,
        }
    )

    config.classifier = "seg"
    config.resnet_pretrained_path = "../model/vit_checkpoint/imagenet21k/R50+ViT-B_16.npz"
    config.decoder_channels = (256, 128, 64, 16)
    config.skip_channels = [512, 256, 64, 16]
    config.n_classes = 2
    config.activation = "softmax"
    return config


def get_l32_config() -> DictConfig:
    """Returns the ViT-L/32 configuration."""
    config = get_l16_config()
    config.patches.size = (32, 32)
    return config


def get_h14_config() -> DictConfig:
    """Returns the ViT-H/14 configuration."""
    return _cfg(
        {
            "patches": {"size": (14, 14)},
            "hidden_size": 1280,
            "transformer": {
                "mlp_dim": 5120,
                "num_heads": 16,
                "num_layers": 32,
                "attention_dropout_rate": 0.0,
                "dropout_rate": 0.1,
            },
            "classifier": "token",
            "representation_size": None,
        }
    )
