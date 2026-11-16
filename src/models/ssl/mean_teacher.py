'''
Mean Teacher (Tarvainen & Valpola, arXiv:1703.01780) for 2D segmentation.

Ported from the reference 2D implementation in HiLab-git/SSL4MIS
(https://github.com/HiLab-git/SSL4MIS, MIT, commit 06df6047a59aba9988ced8331998b5957ecb356b),
file `code/train_mean_teacher_2D.py`. Behaviour kept faithful to that script:

  - the teacher is an EMA copy of the student with detached parameters (`ema_decay=0.99`), updated
    with the "true average until the exponential average is more correct" rule
    `alpha = min(1 - 1/(step+1), ema_decay)`;
  - the teacher is fed **only the unlabeled rows**, perturbed with `clamp(randn * 0.1, -0.2, 0.2)`;
  - the consistency term is the MSE between student and teacher softmax **on those same rows**;
  - the weight follows the same sigmoid ramp, held at 0 during an initial warm-up (both configured
    on `SemiSupervisedLitWrapper`, as *fractions of the training budget* rather than the reference's
    hard-coded step counts -- see the note on `REFERENCE_TOTAL_STEPS` there for why).

The EMA update itself is rewritten: the original uses `add_(scalar, tensor)`, an overload removed
in torch 2.x.
'''

import copy
from collections import deque

import torch
from torch import Tensor, nn

from .base import SemiSupervisedLitWrapper
from ...data.augmentations import gaussian_noise

from typing import Dict, Iterable


BN_BUFFER_MODES = ("copy", "ema", "recalibrate")


@torch.no_grad()
def recalibrate_batchnorm(network: nn.Module, batches: Iterable[Tensor]) -> int:
    '''
    Re-estimate every BatchNorm's running mean/var of `network` **in place** from `batches`.

    Running stats are reset and the momentum set to None (cumulative average, so every batch weighs
    the same), the network is run in train mode without gradients, and then everything -- momentum
    and train/eval mode -- is restored. Parameters are never touched: only the buffers change.

    Returns the number of batches used (0 leaves the network as it was, stats included).
    '''
    batches = list(batches)
    bn_layers = [m for m in network.modules() if isinstance(m, nn.modules.batchnorm._BatchNorm)]
    if not batches or not bn_layers:
        return 0
    was_training = network.training
    momenta = [m.momentum for m in bn_layers]
    for m in bn_layers:
        m.reset_running_stats()
        m.momentum = None
    network.train()
    device = next(network.parameters()).device
    for images in batches:
        network(images.to(device))
    for m, momentum in zip(bn_layers, momenta):
        m.momentum = momentum
    network.train(was_training)
    return len(batches)


class MeanTeacherLitWrapper(SemiSupervisedLitWrapper):

    def __init__(self, *args, ema_decay: float = 0.99,
                 noise_std: float = 0.1, noise_clamp: float = 0.2,
                 evaluate_with_teacher: bool = True,
                 bn_buffer_mode: str = "copy",
                 bn_recalibration_batches: int = 32,
                 **kwargs):
        '''
        Args:
            ema_decay: teacher EMA decay (SSL4MIS `--ema_decay`, default 0.99).
            noise_std / noise_clamp: teacher input perturbation.
            evaluate_with_teacher: run val/test through the teacher, which is the usual Mean Teacher
                reporting convention (the EMA weights are the better model).
            bn_buffer_mode: how the teacher's BatchNorm running stats are produced.
                `copy` (default, SSL4MIS): copied from the student every step -- so in eval mode the
                teacher combines AVERAGED weights with the student's CURRENT statistics.
                `ema`: averaged with the same alpha as the weights (timm `ModelEmaV2`), so weights and
                statistics come from the same average.
                `recalibrate`: copied as in `copy` during training, and re-estimated from the last
                `bn_recalibration_batches` training batches right before every validation/test.
            bn_recalibration_batches: size of the ring buffer of recent training batches used by
                `recalibrate` (images only, kept on CPU).
        '''
        if bn_buffer_mode not in BN_BUFFER_MODES:
            raise ValueError(f"bn_buffer_mode deve ser um de {BN_BUFFER_MODES}, recebeu '{bn_buffer_mode}'")
        super().__init__(*args, **kwargs)
        self.save_hyperparameters({
            "ema_decay": ema_decay,
            "noise_std": noise_std,
            "noise_clamp": noise_clamp,
            "evaluate_with_teacher": evaluate_with_teacher,
            "bn_buffer_mode": bn_buffer_mode,
            "bn_recalibration_batches": bn_recalibration_batches,
        })
        self.ema_decay = ema_decay
        self.noise_std = noise_std
        self.noise_clamp = noise_clamp
        self.evaluate_with_teacher = evaluate_with_teacher
        self.bn_buffer_mode = bn_buffer_mode
        self.recent_train_images: deque = deque(maxlen=max(int(bn_recalibration_batches), 1))

        self.teacher = copy.deepcopy(self.model)
        for param in self.teacher.parameters():
            param.detach_()
            param.requires_grad_(False)

    def ema_alpha(self) -> float:
        return min(1.0 - 1.0 / (self.global_step + 1), self.ema_decay)

    @torch.no_grad()
    def update_teacher(self) -> None:
        '''EMA update, rewritten for the torch 2.x `add_(tensor, alpha=...)` signature.'''
        alpha = self.ema_alpha()
        for teacher_param, student_param in zip(self.teacher.parameters(), self.model.parameters()):
            teacher_param.data.mul_(alpha).add_(student_param.data, alpha=1.0 - alpha)
        for teacher_buffer, student_buffer in zip(self.teacher.buffers(), self.model.buffers()):
            if self.bn_buffer_mode == "ema" and teacher_buffer.is_floating_point():
                teacher_buffer.data.mul_(alpha).add_(student_buffer.data, alpha=1.0 - alpha)
            else:
                # `copy`/`recalibrate`, and integer buffers (`num_batches_tracked`) in every mode:
                # copied, not averaged -- as in the reference.
                teacher_buffer.data.copy_(student_buffer.data)

    def recalibrate_teacher_bn(self) -> int:
        '''Re-estimate the teacher's BN statistics from the buffered recent training batches.'''
        return recalibrate_batchnorm(self.teacher, self.recent_train_images)

    def _maybe_recalibrate(self) -> None:
        if self.bn_buffer_mode == "recalibrate" and self.evaluate_with_teacher:
            self.recalibrate_teacher_bn()
            # Leave the teacher in eval mode for the evaluation loop that follows.
            self.teacher.eval()

    def on_validation_epoch_start(self):
        self._maybe_recalibrate()

    def on_test_epoch_start(self):
        self._maybe_recalibrate()

    def unsupervised_loss(self, batch: Dict, logits: Tensor, is_labeled: Tensor) -> Tensor:
        unlabeled = ~is_labeled
        unlabeled_images = batch["image"][unlabeled]

        teacher_inputs = gaussian_noise(unlabeled_images, std=self.noise_std, clamp=self.noise_clamp)
        with torch.no_grad():
            teacher_soft = torch.softmax(self.teacher(teacher_inputs), dim=1)
        self.count_forward("teacher", teacher_inputs.shape[0])

        # The teacher only perturbs intensities -- no geometry -- so its output is already aligned
        # with the hidden ground truth and can be scored against it directly.
        hidden, has_hidden = self.hidden_targets(batch, unlabeled)
        self.log_unlabeled_diagnostics(teacher_soft.argmax(dim=1), hidden, has_hidden,
                                       prefix="train/teacher")

        student_soft = torch.softmax(logits[unlabeled], dim=1)
        return torch.mean((student_soft - teacher_soft) ** 2)

    def on_train_batch_end(self, outputs, batch, batch_idx):
        self.update_teacher()
        if self.bn_buffer_mode == "recalibrate":
            self.recent_train_images.append(batch["image"].detach().cpu())

    def forward(self, x):
        '''Student forward. Evaluation is routed through the teacher by the eval hooks below.'''
        return self.model(x)

    def _eval_forward(self, x):
        return self.teacher(x) if self.evaluate_with_teacher else self.model(x)

    def validation_step(self, batch, batch_idx):
        return self._eval_step(batch, stage="val")

    def test_step(self, batch, batch_idx):
        return self._eval_step(batch, stage="test")

    def _eval_step(self, batch, stage: str):
        x, y = batch["image"], batch["segmentation"]
        y_pred = self._eval_forward(x)
        loss, loss_dict = self.calculate_loss_and_metrics(y_pred, y, stage)
        self.log_dict(loss_dict, prog_bar=True, logger=True, on_step=False, on_epoch=True)
        return loss
