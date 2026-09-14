'''
The latent-diffusion subset DiffRect actually uses.

Provenance
----------
Reimplemented from:
  - DiffRect, Liu, Li & Yuan, "DiffRect: Latent Diffusion Label Rectification for Semi-supervised
    Medical Image Segmentation", MICCAI 2024 (arXiv:2407.09918).
    Reference implementation: https://github.com/CUHK-AIM-Group/DiffRect (MIT), file
    `networks/unet_de.py`, class `DiffUNet`.
  - openai/guided-diffusion (MIT), which DiffRect vendors wholesale under
    `networks/guided_diffusion/` and calls into for `get_named_beta_schedule`, `SpacedDiffusion`,
    `space_timesteps` and `UniformSampler`.

**Reimplementation, not a vendoring.** guided-diffusion is ~50 KB of `ModelMeanType` /
`ModelVarType` / `LossType` / respacing machinery, of which DiffRect reaches perhaps eighty lines:
it instantiates `SpacedDiffusion` with `ModelMeanType.START_X`, `ModelVarType.FIXED_LARGE` and
`LossType.MSE`, and then only ever calls `q_sample` and `ddim_sample_loop`. Carrying the rest --
including several code paths that are dead under that configuration -- would make it harder, not
easier, to see what the method does. What is kept here is arithmetically identical; the parity
tests in `tests/test_reference_equivalence.py` pin that against transcribed upstream snippets.

Consequences of the reference's configuration, made explicit because they look like omissions:
  - `START_X` means the denoising network predicts the **clean latent x0 directly**, not the noise.
    So there is no `eps -> x0` conversion anywhere below, and the training loss is a plain MSE
    between the prediction and the clean latent.
  - `FIXED_LARGE` means the reverse-process variance is a fixed schedule constant rather than a
    learned head. The model therefore emits no variance channels and none are read.
  - Sampling is DDIM with `eta=0` (deterministic), respaced from `timesteps` to `sample_steps`.
'''

import math

import torch
from torch import Tensor, nn

from typing import Callable, List, Optional


def cosine_beta_schedule(timesteps: int, s: float = 0.008, max_beta: float = 0.999) -> Tensor:
    '''
    Cosine noise schedule from Nichol & Dhariwal, "Improved Denoising Diffusion Probabilistic
    Models" (arXiv:2102.09672), in the form guided-diffusion's `get_named_beta_schedule("cosine")`
    produces: betas derived from `alpha_bar(t) = cos((t/T + s)/(1 + s) * pi/2)^2`, clipped at
    `max_beta` to keep the last step from being a singularity.

    DiffRect passes `--ldm_beta_sch cosine` and `--ts 10`, so this is evaluated at T=10.
    '''
    def alpha_bar(t: float) -> float:
        return math.cos((t + s) / (1.0 + s) * math.pi / 2.0) ** 2

    betas = []
    for i in range(timesteps):
        t1, t2 = i / timesteps, (i + 1) / timesteps
        betas.append(min(1.0 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return torch.tensor(betas, dtype=torch.float64)


def linear_beta_schedule(timesteps: int) -> Tensor:
    '''
    guided-diffusion's `"linear"` schedule, scaled so its endpoints stay meaningful away from the
    T=1000 it was tuned at. Not used by DiffRect's ACDC configuration; kept so `beta_schedule` is
    a real choice and the cosine default is visibly a choice.
    '''
    scale = 1000.0 / timesteps
    return torch.linspace(scale * 1e-4, scale * 2e-2, timesteps, dtype=torch.float64)


BETA_SCHEDULES = {"cosine": cosine_beta_schedule, "linear": linear_beta_schedule}


def _space_timesteps(num_timesteps: int, section_count: int) -> List[int]:
    '''
    guided-diffusion's `space_timesteps(num_timesteps, [section_count])` for the single-section
    case DiffRect uses (`space_timesteps(ts, [ts_sample])`).

    Picks `section_count` timesteps out of `num_timesteps` with as even a stride as integer
    arithmetic allows. With DiffRect's `ts=10, ts_sample=2` this is `[0, 5]`.
    '''
    if section_count > num_timesteps:
        raise ValueError(f"cannot take {section_count} steps out of {num_timesteps}")
    if section_count <= 1:
        return [0]
    stride = (num_timesteps - 1) / (section_count - 1)
    taken, current = [], 0.0
    for _ in range(section_count):
        taken.append(round(current))
        current += stride
    return taken


class LatentDiffusion(nn.Module):
    '''
    Forward (noising) and reverse (DDIM sampling) processes over DiffRect's label latents.

    Holds only buffers, no parameters: the denoising network is passed in as a callable, because
    DiffRect conditions it on a second latent that this class knows nothing about.
    '''

    def __init__(self, timesteps: int = 10, sample_steps: int = 2, schedule: str = "cosine",
                 clip_denoised: bool = True):
        '''
        Args:
            clip_denoised: clamp the predicted clean latent to [-1, 1] at every sampling step.

                **True reproduces the reference, and is almost certainly not what its authors
                intended.** `ddim_sample_loop(clip_denoised=True)` is guided-diffusion's default and
                DiffRect never overrides it -- but that default exists because guided-diffusion
                samples *images* in [-1, 1], while what is being sampled here is a 256-channel
                feature map out of a BatchNorm + LeakyReLU encoder, whose activations are not
                bounded by 1. So the clamp is a real, load-bearing nonlinearity in the sampling
                path, inherited from an image-space assumption.

                It is kept True because it is what produced the published numbers, and exposed as a
                flag because "does removing it help?" is a one-line ablation worth running.
        '''
        super().__init__()
        if schedule not in BETA_SCHEDULES:
            raise ValueError(f"Unknown beta schedule {schedule!r}; expected one of {sorted(BETA_SCHEDULES)}.")
        if not 1 <= sample_steps <= timesteps:
            raise ValueError(f"sample_steps must be in [1, timesteps]; got {sample_steps} and {timesteps}.")

        self.timesteps = timesteps
        self.sample_steps = sample_steps
        self.schedule = schedule
        self.clip_denoised = clip_denoised

        betas = BETA_SCHEDULES[schedule](timesteps)
        alphas_cumprod = torch.cumprod(1.0 - betas, dim=0)

        self.register_buffer("betas", betas.float(), persistent=False)
        self.register_buffer("alphas_cumprod", alphas_cumprod.float(), persistent=False)
        self.register_buffer("sqrt_alphas_cumprod", alphas_cumprod.sqrt().float(), persistent=False)
        self.register_buffer("sqrt_one_minus_alphas_cumprod",
                             (1.0 - alphas_cumprod).sqrt().float(), persistent=False)
        self.register_buffer("sample_timesteps_index",
                             torch.tensor(_space_timesteps(timesteps, sample_steps), dtype=torch.long),
                             persistent=False)

    # ------------------------------------------------------------------ forward process

    def sample_timesteps(self, batch_size: int, device, generator: Optional[torch.Generator] = None) -> Tensor:
        '''Uniform timesteps, matching guided-diffusion's `UniformSampler` (weights are all 1, so
        the importance weight it also returns is always 1 and the reference discards it).'''
        return torch.randint(0, self.timesteps, (batch_size,), device=device, generator=generator)

    def q_sample(self, x_start: Tensor, t: Tensor, noise: Optional[Tensor] = None) -> Tensor:
        '''`q(x_t | x_0) = N(sqrt(alpha_bar_t) x_0, (1 - alpha_bar_t) I)`, sampled.'''
        if noise is None:
            noise = torch.randn_like(x_start)
        shape = (-1,) + (1,) * (x_start.dim() - 1)
        sqrt_ab = self.sqrt_alphas_cumprod[t].view(shape)
        sqrt_1mab = self.sqrt_one_minus_alphas_cumprod[t].view(shape)
        return sqrt_ab * x_start + sqrt_1mab * noise

    # ------------------------------------------------------------------ reverse process

    @torch.no_grad()
    def ddim_sample(
        self,
        denoise_fn: Callable[[Tensor, Tensor], Tensor],
        shape,
        device,
        generator: Optional[torch.Generator] = None,
        eta: float = 0.0,
        clip_denoised: Optional[bool] = None,
    ) -> Tensor:
        '''
        Deterministic DDIM sampling from pure noise, returning the final predicted clean latent.

        This is Eq. 12 of the paper: the rectified label feature is generated by starting from
        `r^T ~ N(0, I)` and progressively denoising with the weak pseudo-label latent as condition.
        The condition is closed over by `denoise_fn`.

        Args:
            denoise_fn: `(x_t, t) -> pred_x_start`. `t` is `[B]` long, in the *unrespaced* range
                `[0, timesteps)`, matching what the training path passes -- guided-diffusion's
                `_WrappedModel` performs the same respaced-to-original mapping.
            shape: shape of the latent to sample, `[B, C, H, W]`.
            eta: 0.0 is deterministic DDIM (what the reference uses). >0 adds the stochastic term.
            clip_denoised: clamp each step's predicted clean latent to [-1, 1]. Defaults to
                `self.clip_denoised` -- see the note there before changing it.
        '''
        if clip_denoised is None:
            clip_denoised = self.clip_denoised

        x_t = torch.randn(shape, device=device, generator=generator)
        indices = self.sample_timesteps_index.tolist()

        for position, step in enumerate(reversed(indices)):
            t = torch.full((shape[0],), step, device=device, dtype=torch.long)
            pred_x_start = denoise_fn(x_t, t)
            if clip_denoised:
                pred_x_start = pred_x_start.clamp(-1.0, 1.0)

            alpha_bar_t = self.alphas_cumprod[step]
            is_last = position == len(indices) - 1
            if is_last:
                x_t = pred_x_start
                break

            prev_step = list(reversed(indices))[position + 1]
            alpha_bar_prev = self.alphas_cumprod[prev_step]

            # Recover the implied noise, then re-noise to the previous (less noisy) timestep.
            eps = (x_t - alpha_bar_t.sqrt() * pred_x_start) / (1.0 - alpha_bar_t).sqrt()
            sigma = eta * (
                ((1.0 - alpha_bar_prev) / (1.0 - alpha_bar_t)).sqrt()
                * (1.0 - alpha_bar_t / alpha_bar_prev).sqrt()
            )
            x_t = (
                alpha_bar_prev.sqrt() * pred_x_start
                + (1.0 - alpha_bar_prev - sigma ** 2).clamp(min=0.0).sqrt() * eps
            )
            if eta > 0:
                x_t = x_t + sigma * torch.randn(shape, device=device, generator=generator)

        return x_t
