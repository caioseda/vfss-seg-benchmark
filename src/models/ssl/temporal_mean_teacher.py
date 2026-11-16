'''
Mean Teacher + consistência temporal transportada (E2, cenários 4 e 5).

    L = L_sup + lambda_MT(s) * L_MT + lambda_temp(s) * L_temp

  - L_sup: 0,5*(CE + Dice) nos frames rotulados (herdado de `SemiSupervisedLitWrapper`).
  - L_MT: MSE entre softmax do STUDENT no central não rotulado t e softmax do TEACHER EMA no mesmo
    frame com ruído (idêntico a `MeanTeacherLitWrapper`).
  - L_temp: para cada vizinho válido n do central t (t-1; ou t-1 e t+1; ou qualquer janela), a
    predição do TEACHER em n é transportada para t por fluxo óptico, P~_{n->t} = W(P_n, F), e o
    STUDENT em t é puxado para ela:

        L_temp = media_n [ sum_x M_noc(x) * mean_c (p_t(x) - P~_{n->t}(x))^2  /  sum_x M_noc(x) ]

    O M_noc (confiança do warp, `src/data/optical_flow.py`) desconta os pixels que o fluxo não
    explica -- oclusão, bolo, fluxo ruim. Cada vizinho pesa igual, independentemente da sua massa de
    confiança; vizinhos inválidos (fora do vídeo) não entram.

O alvo é sempre o teacher; o student é a única rede otimizada. Os vizinhos só passam pelo teacher
(`no_grad`): o student nunca os vê, como na tabela do pedido.

Os dois termos não supervisionados têm cronogramas próprios. `lambda_MT` é o herdado (warm-up + rampa
sigmoide ancorados no orçamento). `lambda_temp` fica em ZERO até `temporal_warmup` e só então começa a
sua rampa sigmoide, de comprimento `temporal_rampup` -- a rampa começa no fim do warm-up, não no passo
0, para que o termo temporal só entre quando o teacher já produz algo que valha transportar.

Os cenários 4 e 5 diferem só em `neighbor_offsets` do dataset (`[-1]` vs `[-1, 1]`); este módulo usa
os K vizinhos que o batch trouxer, então uma janela maior não exige mudança aqui.
'''

from typing import Dict, List, Optional, Tuple

import torch
from torch import Tensor

from .base import sigmoid_rampup
from .mean_teacher import MeanTeacherLitWrapper
from ...data.augmentations import gaussian_noise, weak_augment_temporal
from ...data.optical_flow import WARP_CONVENTIONS, warp

NEIGHBOR_KEYS = ("neighbors", "neighbor_flow", "neighbor_confidence", "neighbor_valid")


class TemporalAugmentMixin:
    '''
    `student_augment` para batches com vizinhos temporais: a mesma geometria do central vai aos vizinhos,
    ao fluxo (reamostrado E com os vetores girados/escalados) e ao M_noc -- `weak_augment_temporal`.
    Compartilhado pelo MT temporal e pelo DTC-TRUS; precisa vir ANTES de `SemiSupervisedLitWrapper` na MRO.
    '''

    def _augmented_keys(self, batch: Dict) -> List[str]:
        keys = super()._augmented_keys(batch)
        if all(key in batch for key in NEIGHBOR_KEYS):
            keys += ["neighbors", "neighbor_flow", "neighbor_confidence"]
        return keys

    def _weak_augment_rows(self, rows: Dict[str, Tensor]) -> Dict[str, Tensor]:
        if "neighbor_flow" not in rows:
            return super()._weak_augment_rows(rows)
        mask_keys = [key for key in ("segmentation", "hidden_segmentation") if key in rows]
        image, masks, neighbors, flow, confidence = weak_augment_temporal(
            rows["image"], [rows[key] for key in mask_keys],
            rows["neighbors"], rows["neighbor_flow"], rows["neighbor_confidence"])
        return {"image": image, **dict(zip(mask_keys, masks)), "neighbors": neighbors,
                "neighbor_flow": flow, "neighbor_confidence": confidence}


class TemporalMeanTeacherLitWrapper(TemporalAugmentMixin, MeanTeacherLitWrapper):

    # `student_augment` leva a mesma geometria do central aos vizinhos, ao fluxo e ao M_noc
    # (`TemporalAugmentMixin`).
    supports_student_augment = True

    def __init__(self, *args,
                 temporal_weight: float = 0.1,
                 temporal_warmup_fraction: float = 0.0333,
                 temporal_rampup_fraction: Optional[float] = None,
                 temporal_warmup_steps: Optional[float] = None,
                 temporal_rampup_steps: Optional[float] = None,
                 temporal_use_confidence: bool = True,
                 warp_convention: str = "dtc_trus",
                 **kwargs):
        '''
        Args:
            temporal_weight: peso máximo de L_temp.
            temporal_warmup_fraction: fração do orçamento com lambda_temp = 0.
            temporal_rampup_fraction: comprimento da rampa, em fração do orçamento, contado a partir
                do fim do warm-up. None = até o último passo (peso máximo exatamente no fim, o mesmo
                formato do cronograma de referência do MT).
            temporal_warmup_steps / temporal_rampup_steps: sobrescritas absolutas, em passos.
            temporal_use_confidence: ponderar por M_noc; False = média simples sobre os pixels.
            warp_convention: tem de ser a MESMA do dataset (`dtc_trus` | `backward`), porque o sinal do
                fluxo depende dela.
        '''
        super().__init__(*args, **kwargs)
        if warp_convention not in WARP_CONVENTIONS:
            raise ValueError(f"warp_convention deve ser uma de {WARP_CONVENTIONS}, recebeu '{warp_convention}'")
        self.save_hyperparameters({
            "temporal_weight": temporal_weight,
            "temporal_warmup_fraction": temporal_warmup_fraction,
            "temporal_rampup_fraction": temporal_rampup_fraction,
            "temporal_warmup_steps": temporal_warmup_steps,
            "temporal_rampup_steps": temporal_rampup_steps,
            "temporal_use_confidence": temporal_use_confidence,
            "warp_convention": warp_convention,
        })
        self.temporal_weight = temporal_weight
        self.temporal_warmup_fraction = temporal_warmup_fraction
        self.temporal_rampup_fraction = temporal_rampup_fraction
        self._temporal_warmup_override = temporal_warmup_steps
        self._temporal_rampup_override = temporal_rampup_steps
        self.temporal_use_confidence = temporal_use_confidence
        self.warp_convention = warp_convention

    # ------------------------------------------------------------------ schedule

    @property
    def temporal_warmup(self) -> float:
        return self.budget_fraction_steps(self.temporal_warmup_fraction, self._temporal_warmup_override)

    @property
    def temporal_rampup(self) -> float:
        if self._temporal_rampup_override is not None:
            return float(self._temporal_rampup_override)
        if self.temporal_rampup_fraction is None:
            return max(self.total_training_steps - self.temporal_warmup, 0.0)
        return self.budget_fraction_steps(self.temporal_rampup_fraction)

    def current_temporal_weight(self) -> float:
        '''0 durante o warm-up; depois, rampa sigmoide que COMEÇA no fim do warm-up.'''
        warmup = self.temporal_warmup
        if self.global_step < warmup:
            return 0.0
        return self.temporal_weight * sigmoid_rampup(self.global_step - warmup, self.temporal_rampup)

    # ------------------------------------------------------------------ losses

    def _teacher_forward(self, centers: Tensor, neighbors: Tensor) -> Tuple[Tensor, Tensor]:
        '''Um único forward do teacher sobre (centrais, vizinhos), cada um com o ruído do MT.'''
        inputs = torch.cat([centers, neighbors], dim=0)
        inputs = gaussian_noise(inputs, std=self.noise_std, clamp=self.noise_clamp)
        with torch.no_grad():
            soft = torch.softmax(self.teacher(inputs), dim=1)
        self.count_forward("teacher", inputs.shape[0])
        return soft[:centers.shape[0]], soft[centers.shape[0]:]

    def temporal_loss(self, student_soft: Tensor, transported: Tensor, confidence: Tensor) -> Tensor:
        '''
        Args:
            student_soft: `[N,C,H,W]` softmax do student no central de cada par.
            transported: `[N,C,H,W]` softmax do teacher no vizinho, já transportado para o central.
            confidence: `[N,1,H,W]` M_noc (ou uns).
        '''
        squared = ((student_soft - transported) ** 2).mean(dim=1, keepdim=True)
        per_pair = (confidence * squared).flatten(1).sum(1) / confidence.flatten(1).sum(1).clamp_min(1e-6)
        return per_pair.mean()

    # ------------------------------------------------------------------ training

    def training_step(self, batch, batch_idx):
        batch = self.augment_batch(batch)
        images, targets = batch["image"], batch["segmentation"]
        is_labeled = self.labeled_mask(batch).to(images.device)
        unlabeled = ~is_labeled

        logits = self.forward(images)
        self.count_forward("student", images.shape[0])

        if is_labeled.any():
            sup_loss = self.supervised_loss(logits[is_labeled], targets[is_labeled])
        else:
            sup_loss = logits.sum() * 0.0

        mt_weight = self.current_consistency_weight()
        temporal_weight = self.current_temporal_weight()
        zero = logits.sum() * 0.0
        mt_loss, temp_loss = zero, zero
        extra: Dict[str, Tensor] = {}

        has_neighbors = all(key in batch for key in NEIGHBOR_KEYS)
        if (mt_weight > 0.0 or temporal_weight > 0.0) and unlabeled.any():
            centers = images[unlabeled]
            student_soft = torch.softmax(logits[unlabeled], dim=1)

            if has_neighbors and temporal_weight > 0.0:
                valid = batch["neighbor_valid"][unlabeled].bool()                 # [U,K]
                neighbor_images = batch["neighbors"][unlabeled][valid]             # [N,C,H,W]
            else:
                valid = torch.zeros((centers.shape[0], 0), dtype=torch.bool, device=images.device)
                neighbor_images = centers[:0]

            teacher_center, teacher_neighbors = self._teacher_forward(centers, neighbor_images)

            hidden, has_hidden = self.hidden_targets(batch, unlabeled)
            self.log_unlabeled_diagnostics(teacher_center.argmax(dim=1), hidden, has_hidden,
                                           prefix="train/teacher")
            mt_loss = torch.mean((student_soft - teacher_center) ** 2)

            if neighbor_images.shape[0] > 0:
                flow = batch["neighbor_flow"][unlabeled][valid]
                transported = warp(teacher_neighbors, flow, self.warp_convention)
                if self.temporal_use_confidence:
                    confidence = batch["neighbor_confidence"][unlabeled][valid].to(transported)
                else:
                    confidence = torch.ones_like(transported[:, :1])
                # Which central each valid (central, neighbour) pair belongs to.
                center_of_pair = torch.arange(valid.shape[0], device=valid.device)[:, None].expand_as(valid)[valid]
                temp_loss = self.temporal_loss(student_soft[center_of_pair], transported, confidence)

                if hidden is not None:
                    self.log_unlabeled_diagnostics(transported.argmax(dim=1), hidden[center_of_pair],
                                                   has_hidden[center_of_pair], prefix="train/transported")
                extra["train/noc_mean"] = confidence.mean().detach()
            extra["train/temporal_valid_frac"] = (valid.float().mean() if valid.numel()
                                                  else torch.zeros((), device=images.device))

        loss = sup_loss + mt_weight * mt_loss + temporal_weight * temp_loss
        extra.update({
            "train/mt_loss": mt_loss.detach(),
            "train/temporal_loss": temp_loss.detach(),
            "train/temporal_loss_weighted": (temporal_weight * temp_loss).detach(),
            "train/temporal_weight": torch.tensor(temporal_weight, device=logits.device),
        })
        # `unsup_loss`/`consistency_weight` keep meaning the MT term, so these runs line up with the
        # plain Mean Teacher's curves key for key.
        self.log_training(loss, sup_loss, mt_loss, mt_weight, is_labeled, logits, targets, extra=extra)
        self.log("train/temporal_weight_step", temporal_weight, prog_bar=False, logger=True,
                 on_step=True, on_epoch=False)
        return loss
