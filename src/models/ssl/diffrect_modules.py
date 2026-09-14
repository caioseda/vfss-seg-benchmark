'''
Building blocks of DiffRect: the Label Context Calibration module (LCC) and the Latent Feature
Rectification module (LFR).

Ported from https://github.com/CUHK-AIM-Group/DiffRect (MIT), `networks/unet_de.py`, where they
live as `UNet_LDMV2` / `DeUNet` / `DiffUNet`. The names here follow the *paper*'s vocabulary rather
than the reference's, because the reference's names ("LDMV2", "good") do not survive being read.

Map from paper to code:

  paper                                  reference                here
  ---------------------------------------------------------------------------------------------
  semantic coloring scheme (SCS)         `pl_weak_embed` et al.   `semantic_coloring`
  semantic context embedding block B_sem `UNet_LDMV2.encoder`     `RectificationNet.encode`
  calibration guidance tau               the `t` argument         `cg` argument / `TimestepEmbedding`
  denoising U-Net epsilon                `DeUNet`                 `LatentDenoiseUNet`
  latent loss L_Lat-U / L_Lat-L          `lat_loss`               second return of `forward_train`

Two places where the reference and the paper disagree, both resolved in favour of the reference
(it is what produced the published numbers), both flagged so the disagreement stays visible:

  1. **The rectifier sees the image.** The paper describes LFR as operating on label latents; the
     reference concatenates the image onto the coloured mask *and* injects multi-scale image
     features into every encoder stage. Only the inner denoising U-Net is purely latent. Exposed as
     `condition_on_image` so the paper's reading is a one-flag ablation.
  2. **Calibration guidance is a Dice *loss*, not a Dice *score*.** Eq. 6 of the paper reads
     `tau = Dice(y_s, y_w)`; the reference computes `dice_loss(...)`, i.e. `1 - Dice`, so high
     guidance means *low* agreement. `calibration_guidance` in `diffrect.py` follows the code.
'''

import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from ...third_party.diffrect import LatentDiffusion

from typing import List, Optional, Sequence, Tuple


# The reference's colour set for ACDC's four classes. The paper motivates the choice as maximising
# pairwise colour distance ("we maximize the color difference between each encoded category to
# avoid semantic confusion"); for C <= 4 that is black plus the primaries, which is what the
# reference hardcodes. `distinct_colors` below generalises it.
CLASS_COLORS_DEFAULT: Tuple[Tuple[int, int, int], ...] = (
    (0, 0, 0),      # background
    (255, 0, 0),
    (0, 255, 0),
    (0, 0, 255),
)

# Latent feature width of the encoder's deepest stage (H/16 x W/16 x 256 in the paper).
ENCODER_CHANNELS = (16, 32, 64, 128, 256)
ENCODER_DROPOUT = (0.05, 0.1, 0.2, 0.3, 0.5)
TIMESTEP_EMBED_DIM = 128
TIMESTEP_HIDDEN_DIM = 512


def distinct_colors(n_classes: int) -> Tensor:
    '''
    `n_classes` RGB colours in [0, 255], background first and black.

    Reproduces `CLASS_COLORS_DEFAULT` exactly for `n_classes <= 4` (so ACDC and the VFSS
    `multiclass_c2_c4` variant are bit-identical to the reference) and falls back to evenly spaced
    maximally saturated hues beyond that, which is the same "maximise the colour difference"
    criterion the paper states.
    '''
    if n_classes <= len(CLASS_COLORS_DEFAULT):
        return torch.tensor(CLASS_COLORS_DEFAULT[:n_classes], dtype=torch.float32)

    colors = [(0.0, 0.0, 0.0)]
    for i in range(n_classes - 1):
        hue = i / (n_classes - 1)
        r, g, b = _hsv_to_rgb(hue, 1.0, 1.0)
        colors.append((r * 255.0, g * 255.0, b * 255.0))
    return torch.tensor(colors, dtype=torch.float32)


def _hsv_to_rgb(h: float, s: float, v: float) -> Tuple[float, float, float]:
    i = int(h * 6.0)
    f = h * 6.0 - i
    p, q, t = v * (1.0 - s), v * (1.0 - f * s), v * (1.0 - (1.0 - f) * s)
    return [(v, t, p), (q, v, p), (p, v, t), (p, q, v), (t, p, v), (v, p, q)][i % 6]


def semantic_coloring(labels: Tensor, colors: Tensor, signed: bool = True) -> Tensor:
    '''
    Semantic coloring scheme (SCS): class indices -> an RGB image.

    The paper's justification is that painting the mask into the visual space lets the rectifier
    reuse the same convolutional machinery it uses on the image, and Table 4 measures the cost of
    removing it (78.28 -> 73.83 Dice, the single largest ablation drop).

    Args:
        labels: `[B, H, W]` class indices.
        colors: `[C, 3]` in [0, 255], from `distinct_colors`.
        signed: return [-1, 1] instead of [0, 1]. **Default True**: this repository's images are
            normalised to [-1, 1] (`VFSSFrameDatasetBase._preprocess_image`), and the coloured mask
            is concatenated onto them. The reference works in [0, 1] for both because SSL4MIS's
            ACDC loader does; concatenating a [0, 1] mask onto a [-1, 1] image would hand the
            encoder two differently-scaled halves.

    Returns:
        `[B, 3, H, W]` float.
    '''
    colors = colors.to(device=labels.device, dtype=torch.float32) / 255.0
    colored = colors[labels.long()]              # [B, H, W, 3]
    colored = colored.permute(0, 3, 1, 2).contiguous()
    return colored * 2.0 - 1.0 if signed else colored


def nearest_color_labels(colored: Tensor, colors: Tensor, signed: bool = True) -> Tensor:
    '''
    Inverse of `semantic_coloring`: nearest colour in `colors` for every pixel.

    Only used to prove the colouring is injective (`tests/test_ssl_methods.py`); the model never
    needs to invert it, because it decodes to logits rather than back to colours.
    '''
    colors = colors.to(device=colored.device, dtype=torch.float32) / 255.0
    if signed:
        colors = colors * 2.0 - 1.0
    # [B, 1, 3, H, W] against [1, C, 3, 1, 1]
    distances = (colored.unsqueeze(1) - colors.view(1, -1, 3, 1, 1)).pow(2).sum(dim=2)
    return distances.argmin(dim=1)


def timestep_embedding(timesteps: Tensor, dim: int) -> Tensor:
    '''
    Sinusoidal embedding, transcribed from the reference's `get_timestep_embedding`
    (`networks/unet_de.py`), which in turn follows DDPM/Fairseq.
    '''
    half_dim = dim // 2
    scale = math.log(10000.0) / (half_dim - 1)
    freqs = torch.exp(torch.arange(half_dim, dtype=torch.float32, device=timesteps.device) * -scale)
    args = timesteps.float()[:, None] * freqs[None, :]
    embedding = torch.cat([torch.sin(args), torch.cos(args)], dim=1)
    if dim % 2 == 1:
        embedding = F.pad(embedding, (0, 1, 0, 0))
    return embedding


def _swish(x: Tensor) -> Tensor:
    return x * torch.sigmoid(x)


class TimestepEmbedding(nn.Module):
    '''
    Turns the calibration guidance into the conditioning vector every `ConvBlock` is offset by.

    This is where the paper's two ideas collapse into one tensor: DiffRect does not have a separate
    "timestep" and "calibration guidance", it *reuses the diffusion timestep channel to carry the
    guidance*. The reference writes this as `t = dice_loss(...) * 999`, feeding a pseudo-timestep
    into a network whose real timesteps live in [0, ts). `scale` reproduces that 999 for the
    default `timesteps=10`, so a guidance of 1.0 (worst possible pseudo-label) maps to the most
    heavily-noised end of the sinusoidal basis.
    '''

    def __init__(self, embed_dim: int = TIMESTEP_EMBED_DIM, hidden_dim: int = TIMESTEP_HIDDEN_DIM,
                 scale: float = 999.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.scale = scale
        self.dense = nn.ModuleList([nn.Linear(embed_dim, hidden_dim), nn.Linear(hidden_dim, hidden_dim)])

    def forward(self, guidance: Tensor, rescale: bool = True) -> Tensor:
        '''
        Args:
            guidance: `[B]`. Calibration guidance in [0, 1] when `rescale`, or a raw diffusion
                timestep in `[0, timesteps)` when not.
        '''
        values = guidance * self.scale if rescale else guidance
        emb = timestep_embedding(values, self.embed_dim)
        return self.dense[1](_swish(self.dense[0](emb)))


class ConvBlock(nn.Module):
    '''Two 3x3 conv + BN + LeakyReLU, with the conditioning vector added between them.'''

    def __init__(self, in_channels: int, out_channels: int, dropout_p: float,
                 embed_dim: int = TIMESTEP_HIDDEN_DIM):
        super().__init__()
        self.conv0 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(),
        )
        self.conv1 = nn.Sequential(
            nn.Dropout(dropout_p),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(),
        )
        self.embed_proj = nn.Linear(embed_dim, out_channels)

    def forward(self, x: Tensor, embedding: Tensor) -> Tensor:
        x = self.conv0(x)
        x = x + self.embed_proj(_swish(embedding))[:, :, None, None]
        return self.conv1(x)


class DownBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dropout_p: float):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.conv = ConvBlock(in_channels, out_channels, dropout_p)

    def forward(self, x: Tensor, embedding: Tensor) -> Tensor:
        return self.conv(self.pool(x), embedding)


class UpBlock(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int, dropout_p: float = 0.0):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, skip_channels, kernel_size=2, stride=2)
        self.conv = ConvBlock(skip_channels * 2, out_channels, dropout_p)

    def forward(self, x: Tensor, skip: Tensor, embedding: Tensor) -> Tensor:
        return self.conv(torch.cat([skip, self.up(x)], dim=1), embedding)


class Encoder(nn.Module):
    '''
    The semantic context embedding block B_sem: five stages, four 2x downsamples, so the latent is
    `H/16 x W/16 x 256` as the paper states.

    `injections` are the multi-scale image features the reference adds stage-by-stage
    (`UNet_LDMV2.embedder`). The paper says "concatenation"; the code adds. Addition it is.
    '''

    def __init__(self, in_channels: int, channels: Sequence[int] = ENCODER_CHANNELS,
                 dropout: Sequence[float] = ENCODER_DROPOUT):
        super().__init__()
        self.channels = tuple(channels)
        self.in_conv = ConvBlock(in_channels, channels[0], dropout[0])
        self.downs = nn.ModuleList([
            DownBlock(channels[i], channels[i + 1], dropout[i + 1]) for i in range(len(channels) - 1)
        ])

    def forward(self, x: Tensor, embedding: Tensor,
                injections: Optional[Sequence[Tensor]] = None) -> List[Tensor]:
        features = [self.in_conv(x, embedding)]
        if injections is not None:
            features[0] = features[0] + injections[0]
        for i, down in enumerate(self.downs):
            feature = down(features[-1], embedding)
            if injections is not None:
                feature = feature + injections[i + 1]
            features.append(feature)
        return features


class Decoder(nn.Module):
    def __init__(self, n_classes: int, channels: Sequence[int] = ENCODER_CHANNELS):
        super().__init__()
        channels = tuple(channels)
        self.ups = nn.ModuleList([
            UpBlock(channels[i + 1], channels[i], channels[i]) for i in reversed(range(len(channels) - 1))
        ])
        self.out_conv = nn.Conv2d(channels[0], n_classes, kernel_size=3, padding=1)

    def forward(self, features: Sequence[Tensor], embedding: Tensor) -> Tensor:
        x = features[-1]
        for i, up in enumerate(self.ups):
            x = up(x, features[-2 - i], embedding)
        return self.out_conv(x)


class LatentDenoiseUNet(nn.Module):
    '''
    The denoising network epsilon: a small U-Net that lives entirely in latent space.

    "The Denoising U-Net down and upsamples the input by 4x, which also uses two 3x3 convolution
    layers per stage" (paper, Sec. 3.1) -- two down blocks, two up blocks over the 256-channel
    latent. The condition (the *other* label distribution's latent) is **added to the input**, which
    is how the reference makes an unconditional architecture conditional.

    Its `embedding` is built from the real diffusion timestep, not from the calibration guidance:
    inside the diffusion process `t` means what it usually means.
    '''

    def __init__(self, channels: int = ENCODER_CHANNELS[-1], hidden: Sequence[int] = (384, 512)):
        super().__init__()
        self.embedding = TimestepEmbedding()
        self.down1 = DownBlock(channels, hidden[0], 0.0)
        self.down2 = DownBlock(hidden[0], hidden[1], 0.0)
        self.up1 = UpBlock(hidden[1], hidden[0], hidden[0])
        self.up2 = UpBlock(hidden[0], channels, channels)

    def forward(self, x_t: Tensor, t: Tensor, condition: Tensor) -> Tensor:
        embedding = self.embedding(t, rescale=False)
        x0 = x_t + condition
        x1 = self.down1(x0, embedding)
        x2 = self.down2(x1, embedding)
        x = self.up1(x2, x1, embedding)
        return self.up2(x, x0, embedding)


class RectificationNet(nn.Module):
    '''
    LCC + LFR as one module: colour a mask, encode it, rectify its latent with latent diffusion,
    decode to logits.

    Two modes, both taking the calibration guidance `cg`:

      - `forward_train(image, colored_input, colored_target, cg)` -- the training path. Encodes the
        *target* (higher-quality) mask to get the clean latent `x0`, noises it, and asks the
        denoiser to recover it **conditioned on the input (lower-quality) mask's latent**. That is
        the "consecutive transportation" the paper builds: strong->weak for unlabeled data, and
        weak->ground-truth for labeled data. Returns `(logits, latent_mse)`.
      - `forward_sample(image, colored_input, cg)` -- inference. Starts from pure noise and DDIM-
        samples the rectified latent conditioned on the input mask's latent, then decodes it.

    `encode`/`decode` are public because experiment X7 (`docs/hipoteses_experimentos.md`) needs the
    label round-trip -- ground truth through the compressor and back -- without the diffusion step.
    '''

    def __init__(
        self,
        image_channels: int,
        n_classes: int,
        latent_channels: int = ENCODER_CHANNELS[-1],
        timesteps: int = 10,
        sample_steps: int = 2,
        beta_schedule: str = "cosine",
        condition_on_image: bool = True,
        clip_denoised: bool = True,
        class_colors: Optional[Sequence[Sequence[int]]] = None,
    ):
        '''
        Args:
            latent_channels: width of the deepest encoder stage -- the label latent the diffusion
                runs in. 256 (the default) is the reference and the paper's stated
                `H/16 x W/16 x 256`; the encoder stages scale proportionally, so 256 gives exactly
                `[16, 32, 64, 128, 256]`.

                Note the reference *appears* to expose this as `--base_chn_rf 64`, but that flag is
                dead code: `train_diffrect_ACDC.py:181` writes it into a `model_dict` that is never
                passed anywhere, and `UNet_LDMV2` hardcodes the widths. 64 is therefore **not** the
                reference value. Made real here because a narrower rectifier is the first knob to
                reach for if the VFSS grid turns out too slow.
            condition_on_image: feed the image to the rectifier (reference behaviour) or keep it
                label-only (the paper's description). See the module docstring.
        '''
        super().__init__()
        colors = torch.tensor(class_colors, dtype=torch.float32) if class_colors is not None \
            else distinct_colors(n_classes)
        if colors.shape[0] < n_classes:
            raise ValueError(f"class_colors has {colors.shape[0]} colours for {n_classes} classes.")
        self.register_buffer("class_colors", colors[:n_classes], persistent=False)

        self.n_classes = n_classes
        self.image_channels = image_channels
        self.condition_on_image = condition_on_image

        scale = latent_channels / ENCODER_CHANNELS[-1]
        channels = tuple(max(4, int(round(c * scale))) for c in ENCODER_CHANNELS)
        self.channels = channels

        # The mask is coloured to 3 channels; the image rides alongside it when conditioning.
        encoder_in = 3 + (image_channels if condition_on_image else 0)
        self.embedding = TimestepEmbedding()
        self.encoder = Encoder(encoder_in, channels)
        self.image_encoder = Encoder(image_channels, channels) if condition_on_image else None
        self.decoder = Decoder(n_classes, channels)
        self.denoiser = LatentDenoiseUNet(channels[-1], hidden=(int(channels[-1] * 1.5), channels[-1] * 2))
        self.diffusion = LatentDiffusion(timesteps=timesteps, sample_steps=sample_steps,
                                         schedule=beta_schedule, clip_denoised=clip_denoised)

    # ------------------------------------------------------------------ LCC

    def color(self, labels: Tensor) -> Tensor:
        '''Class indices `[B, H, W]` -> `[B, 3, H, W]` in [-1, 1]. The SCS of the paper.'''
        return semantic_coloring(labels, self.class_colors, signed=True)

    def encode(self, image: Optional[Tensor], colored_mask: Tensor, cg: Tensor) -> Tuple[List[Tensor], Tensor]:
        '''
        Coloured mask (+ image) -> the encoder's five feature maps, deepest last.

        Returns `(features, embedding)`; the embedding is returned so `decode` can reuse it instead
        of recomputing the sinusoidal projection.
        '''
        embedding = self.embedding(cg)
        if self.condition_on_image:
            if image is None:
                raise ValueError("condition_on_image=True but no image was given.")
            injections = self.image_encoder(image, embedding)
            x = torch.cat([image, colored_mask], dim=1)
        else:
            injections = None
            x = colored_mask
        return self.encoder(x, embedding, injections), embedding

    def decode(self, features: Sequence[Tensor], embedding: Tensor) -> Tensor:
        '''Encoder features -> class logits `[B, C, H, W]`.'''
        return self.decoder(features, embedding)

    # ------------------------------------------------------------------ LFR

    def forward_train(self, image: Optional[Tensor], colored_input: Tensor,
                      colored_target: Tensor, cg: Tensor) -> Tuple[Tensor, Tensor]:
        '''
        Returns `(logits, latent_mse)`.

        `colored_target` is the higher-quality end of the transportation (ground truth for labeled
        rows, the weak pseudo-label for unlabeled ones) and `colored_input` the lower-quality end.
        The target latent is **detached**: the denoiser is trained to reach it, not to move it. A
        missing detach here turns the latent loss into a collapse objective that still trains and
        still logs a falling curve -- pinned by a test.
        '''
        features, embedding = self.encode(image, colored_input, cg)

        with torch.no_grad():
            target_features, _ = self.encode(image, colored_target, cg)
        x_start = target_features[-1].detach()

        t = self.diffusion.sample_timesteps(x_start.shape[0], x_start.device)
        x_t = self.diffusion.q_sample(x_start, t)
        pred_x_start = self.denoiser(x_t, t, condition=features[-1])
        latent_mse = F.mse_loss(pred_x_start, x_start)

        # `ldm_method='replace'` in the reference: the rectified latent *becomes* the deepest
        # feature, so the decoder reads the corrected distribution rather than the original one.
        features = list(features)
        features[-1] = pred_x_start
        return self.decode(features, embedding), latent_mse

    def forward_sample(self, image: Optional[Tensor], colored_input: Tensor, cg: Tensor,
                       generator: Optional[torch.Generator] = None) -> Tensor:
        '''
        Rectify a pseudo-label: sample the corrected latent from noise, conditioned on the input
        mask's latent, and decode it. Eq. 12 of the paper.
        '''
        features, embedding = self.encode(image, colored_input, cg)
        condition = features[-1]

        sampled = self.diffusion.ddim_sample(
            denoise_fn=lambda x_t, t: self.denoiser(x_t, t, condition=condition),
            shape=condition.shape,
            device=condition.device,
            generator=generator,
        )
        features = list(features)
        features[-1] = sampled
        return self.decode(features, embedding)
