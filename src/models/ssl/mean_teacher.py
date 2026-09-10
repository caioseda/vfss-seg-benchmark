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

import torch
from torch import Tensor

from .base import SemiSupervisedLitWrapper
from ...data.augmentations import gaussian_noise

from typing import Dict


class MeanTeacherLitWrapper(SemiSupervisedLitWrapper):

    def __init__(self, *args, ema_decay: float = 0.99,
                 noise_std: float = 0.1, noise_clamp: float = 0.2,
                 evaluate_with_teacher: bool = True, **kwargs):
        '''
        Args:
            ema_decay: teacher EMA decay (SSL4MIS `--ema_decay`, default 0.99).
            noise_std / noise_clamp: teacher input perturbation.
            evaluate_with_teacher: run val/test through the teacher, which is the usual Mean Teacher
                reporting convention (the EMA weights are the better model).
        '''
        super().__init__(*args, **kwargs)
        self.save_hyperparameters({
            "ema_decay": ema_decay,
            "noise_std": noise_std,
            "noise_clamp": noise_clamp,
            "evaluate_with_teacher": evaluate_with_teacher,
        })
        self.ema_decay = ema_decay
        self.noise_std = noise_std
        self.noise_clamp = noise_clamp
        self.evaluate_with_teacher = evaluate_with_teacher

        self.teacher = copy.deepcopy(self.model)
        for param in self.teacher.parameters():
            param.detach_()
            param.requires_grad_(False)

    @torch.no_grad()
    def update_teacher(self) -> None:
        '''EMA update, rewritten for the torch 2.x `add_(tensor, alpha=...)` signature.'''
        alpha = min(1.0 - 1.0 / (self.global_step + 1), self.ema_decay)
        for teacher_param, student_param in zip(self.teacher.parameters(), self.model.parameters()):
            teacher_param.data.mul_(alpha).add_(student_param.data, alpha=1.0 - alpha)
        # Buffers (BatchNorm running stats) are copied, not averaged -- as in the reference.
        for teacher_buffer, student_buffer in zip(self.teacher.buffers(), self.model.buffers()):
            teacher_buffer.data.copy_(student_buffer.data)

    def unsupervised_loss(self, batch: Dict, logits: Tensor, is_labeled: Tensor) -> Tensor:
        unlabeled = ~is_labeled
        unlabeled_images = batch["image"][unlabeled]

        teacher_inputs = gaussian_noise(unlabeled_images, std=self.noise_std, clamp=self.noise_clamp)
        with torch.no_grad():
            teacher_soft = torch.softmax(self.teacher(teacher_inputs), dim=1)

        # The teacher only perturbs intensities -- no geometry -- so its output is already aligned
        # with the hidden ground truth and can be scored against it directly.
        hidden, has_hidden = self.hidden_targets(batch, unlabeled)
        self.log_unlabeled_diagnostics(teacher_soft.argmax(dim=1), hidden, has_hidden,
                                       prefix="train/teacher")

        student_soft = torch.softmax(logits[unlabeled], dim=1)
        return torch.mean((student_soft - teacher_soft) ** 2)

    def on_train_batch_end(self, outputs, batch, batch_idx):
        self.update_teacher()

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
