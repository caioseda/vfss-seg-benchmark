'''
DTC-TRUS (Distilling Temporal Coherence into 2D Networks for TRUS Prostate Video Segmentation, MICCAI
2026; github.com/DYDevelop/DTC-TRUS) -- a ETAPA 2, adaptada ao E2.

O método tem duas etapas:

  1. um segmentador supervisionado (o "static-image teacher");
  2. um aluno 2D, treinado SEM rótulos nos vídeos, com o modelo da etapa 1 CONGELADO como professor:

        L = 3 * L_seg + 1 * L_KD + 2 * L_con + 0.1 * L_proto       (pesos fixos, sem rampa)

     - L_seg: perda de segmentação contra um pseudo-rótulo SUAVE por flip -- o próprio aluno, em eval e
       sem gradiente, prediz a imagem espelhada na horizontal; a saída é desespelhada e vira o alvo.
     - L_KD: MSE(p_s^T, p_t^T) * T^2, com p^T = probabilidades na temperatura T (T=4).
     - L_con: 1 - mean_x[ M_noc(x) * (1 - |P_t(x) - W(P_{t-1})(x)|) ]. As DUAS predições são do aluno e
       nenhuma é destacada: o gradiente passa pelo frame atual e pelo anterior.
     - L_proto: alinhamento de protótipos entre t-1 e t. Local (FG, features do último mapa antes da cabeça,
       transportadas pelo fluxo, na interseção FG atual x FG anterior transportado) e global (BG, features
       do gargalo, sem fluxo). FG/BG por um limiar adaptativo (percentil 70 das probabilidades FG do frame
       atual, média do lote, sem gradiente). Combinação do original:
           L_proto = w_bg * L_local + w_fg * L_global,   w_fg = fração FG suave média,  w_bg = 1 - w_fg
       -- os pesos parecem TROCADOS em relação ao comentário do código original (o termo FG é pesado pela
       fração BG); reproduzimos como está.

No E2 a etapa 1 é a célula `supervised` pareada (mesma seed/arquitetura/política/fold), lida pelo sweep
(`src/sweeps/e2.py`, `teacher_from`) e passada aqui em `teacher_checkpoint`. O aluno NÃO herda os pesos do
professor (como no original): parte do ImageNet.

Fidelidade e adaptações (decididas para o E2):
  - a etapa 2 NÃO usa GT: os 16 frames do lote 8+8 são tratados como não rotulados. A GT dos rotulados só
    aparece em diagnósticos observacionais (`train/flip_pl_*`, `train/teacher_*`, `train/dice_score`);
  - 2 classes com softmax em vez de 1 canal com sigmoid: KD em softmax(z/T); L_seg = 0,5 * (CE + Dice)
    suaves (a composição do protocolo) contra o alvo suave; |P_t - W(P_n)| é a média sobre as classes
    (= |Δp_FG| com 2 classes); FG = 1 - p(fundo);
  - um par (t-1 -> t) por central (`neighbor_offsets: [-1]`), em vez de sequências de 3 frames com 2 pares;
  - L_seg e L_KD nos centrais (paridade de imagens por passo com os outros métodos); os vizinhos passam pelo
    aluno, COM gradiente, só para L_con e L_proto -- no mesmo forward dos centrais, como o original, que
    empilha os B*T frames;
  - features locais da DeepLabV3+/FPN estão a 1/4 da resolução: as máscaras FG/BG são reduzidas por
    vizinho mais próximo e o fluxo é reamostrado com os vetores escalados;
  - otimizador, LR, passos e `student_augment` são os do protocolo comum do E2 (não o lr=1e-3 com
    `CosineAnnealingWarmRestarts` e o grad-clip do original).
'''

from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf
from torch import Tensor

from .base import SemiSupervisedLitWrapper
from .prototype import (  # noqa: F401  (reexportados: os testes os importam daqui)
    PrototypeLossMixin, adaptive_threshold, masked_avg_pool, resize_flow, resize_mask)
from .temporal_mean_teacher import NEIGHBOR_KEYS, TemporalAugmentMixin
from ...data.optical_flow import WARP_CONVENTIONS, warp
from ...utils import instantiate_from_config


def _plain(value):
    return OmegaConf.to_container(value, resolve=True) if isinstance(value, DictConfig) else value


# Chaves do model_cfg que só definem a INICIALIZAÇÃO (pesos pré-treinados), não a arquitetura: o state_dict do
# professor as sobrescreve, então não entram na comparação (o recarregamento para avaliação as anula).
INIT_ONLY_MODEL_KEYS = ("encoder_weights", "pretrain_ckpt")


def _architecture(model_cfg):
    cfg = _plain(model_cfg)
    if isinstance(cfg, dict) and isinstance(cfg.get("params"), dict):
        cfg = {**cfg, "params": {k: v for k, v in cfg["params"].items() if k not in INIT_ONLY_MODEL_KEYS}}
    return cfg


def load_teacher_state_dict(checkpoint_path: str, expected_model_cfg=None) -> Dict[str, Tensor]:
    '''
    Pesos da REDE de um checkpoint Lightning de um `LitWrapper` (as chaves `model.*`, sem o prefixo).

    Se o checkpoint registrou `model_cfg` nos hiperparâmetros e `expected_model_cfg` for dado, as arquiteturas
    têm de coincidir: um professor de outra arquitetura carregaria (ou falharia) por acidente. Chaves só de
    inicialização (`INIT_ONLY_MODEL_KEYS`) são ignoradas.
    '''
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    recorded = (checkpoint.get("hyper_parameters") or {}).get("model_cfg")
    if (expected_model_cfg is not None and recorded is not None
            and _architecture(recorded) != _architecture(expected_model_cfg)):
        raise ValueError(
            f"O professor em {checkpoint_path} foi treinado com outro `model_cfg`:\n"
            f"  checkpoint: {_plain(recorded)}\n  esperado:   {_plain(expected_model_cfg)}"
        )
    state = {key[len("model."):]: value for key, value in checkpoint["state_dict"].items()
             if key.startswith("model.")}
    if not state:
        raise ValueError(f"{checkpoint_path} não tem chaves `model.*` no state_dict.")
    return state


class DTCTRUSLitWrapper(PrototypeLossMixin, TemporalAugmentMixin, SemiSupervisedLitWrapper):

    # `student_augment` vale para o lote inteiro (não há linhas "rotuladas" para o método) e leva a mesma
    # geometria aos vizinhos, ao fluxo e ao M_noc (`TemporalAugmentMixin`).
    supports_student_augment = True
    student_augment_rows = "all"

    def __init__(self, *args,
                 teacher_checkpoint: Optional[str] = None,
                 teacher_run_id: Optional[str] = None,
                 seg_weight: float = 3.0,
                 kd_weight: float = 1.0,
                 kd_temperature: float = 4.0,
                 con_weight: float = 2.0,
                 proto_weight: float = 0.1,
                 use_prototype: bool = True,
                 proto_percentile: float = 0.7,
                 warp_convention: str = "dtc_trus",
                 **kwargs):
        '''
        Args:
            teacher_checkpoint: checkpoint Lightning da etapa 1 (no E2, o best.ckpt da célula
                `supervised` pareada, preenchido pelo sweep). Obrigatório.
            teacher_run_id: só proveniência (`run_id` da célula da etapa 1).
            seg_weight / kd_weight / con_weight / proto_weight: pesos fixos (padrão do original: 3/1/2/0.1).
            kd_temperature: T da destilação.
            use_prototype: liga L_proto (exige `forward_features` na rede: as arquiteturas smp).
            proto_percentile: percentil do limiar adaptativo FG/BG.
            warp_convention: a MESMA do dataset (`dtc_trus` | `backward`).
        '''
        super().__init__(*args, **kwargs)
        if warp_convention not in WARP_CONVENTIONS:
            raise ValueError(f"warp_convention deve ser uma de {WARP_CONVENTIONS}, recebeu '{warp_convention}'")
        if not teacher_checkpoint:
            raise ValueError(
                "DTCTRUSLitWrapper precisa de `teacher_checkpoint` (a etapa 1). No sweep do E2 ele vem da "
                "célula `supervised` pareada; rodado sozinho: "
                "--set lit_module.params.teacher_checkpoint=<caminho do best.ckpt supervisionado>."
            )
        if use_prototype and not hasattr(self.model, "forward_features"):
            raise ValueError(
                f"use_prototype=True exige `forward_features` em {type(self.model).__name__} (as "
                "arquiteturas smp têm); use use_prototype=False nas demais."
            )
        self.save_hyperparameters({
            "teacher_checkpoint": str(teacher_checkpoint),
            "teacher_run_id": teacher_run_id,
            "seg_weight": seg_weight,
            "kd_weight": kd_weight,
            "kd_temperature": kd_temperature,
            "con_weight": con_weight,
            "proto_weight": proto_weight,
            "use_prototype": use_prototype,
            "proto_percentile": proto_percentile,
            "warp_convention": warp_convention,
        })
        self.teacher_checkpoint = str(teacher_checkpoint)
        self.seg_weight = seg_weight
        self.kd_weight = kd_weight
        self.kd_temperature = kd_temperature
        self.con_weight = con_weight
        self.proto_weight = proto_weight
        self.use_prototype = use_prototype
        self.proto_percentile = proto_percentile
        self.warp_convention = warp_convention

        # Submódulo (vai no state_dict e no `ResourceMonitor`), mas fora do otimizador:
        # `optimizer_param_groups` devolve só `self.model.parameters()`.
        self.teacher = instantiate_from_config(self.model_cfg)
        self.teacher.load_state_dict(load_teacher_state_dict(self.teacher_checkpoint, self.model_cfg), strict=True)
        self.teacher.eval()
        for param in self.teacher.parameters():
            param.requires_grad_(False)

    def train(self, mode: bool = True):
        '''O professor fica em eval sempre (BN com as estatísticas da etapa 1), como no original.'''
        super().train(mode)
        self.teacher.eval()
        return self

    # ------------------------------------------------------------------ losses

    def soft_segmentation_loss(self, logits: Tensor, target: Tensor) -> Tensor:
        '''`0.5 * (CE + Dice)` contra um alvo SUAVE `[N,C,H,W]` -- o `supervised_loss` com probabilidades.'''
        ce = -(target * F.log_softmax(logits, dim=1)).sum(dim=1).mean()
        probs = torch.softmax(logits, dim=1)
        dice = torch.stack([self.dice_loss._dice_loss(probs[:, c], target[:, c])
                            for c in range(probs.shape[1])]).mean()
        return 0.5 * (ce + dice)

    def kd_loss(self, student_logits: Tensor, teacher_logits: Tensor) -> Tensor:
        T = self.kd_temperature
        return F.mse_loss(torch.softmax(student_logits / T, dim=1),
                          torch.softmax(teacher_logits / T, dim=1)) * T * T

    @staticmethod
    def temporal_consistency_loss(current: Tensor, transported: Tensor, confidence: Tensor) -> Tensor:
        '''`1 - mean(M_noc * (1 - |P_t - W(P_n)|))`, média sobre pares e pixels (não normalizada por M_noc).'''
        diff = (current - transported).abs().mean(dim=1, keepdim=True)
        return 1.0 - (confidence * (1.0 - diff)).mean()

    # ------------------------------------------------------------------ training

    @torch.no_grad()
    def flip_pseudo_label(self, images: Tensor) -> Tensor:
        '''Alvo suave por flip horizontal: o aluno em eval sobre a imagem espelhada, desespelhado.'''
        was_training = self.model.training
        self.model.eval()
        try:
            probs = torch.softmax(self.model(torch.flip(images, dims=[-1])), dim=1)
        finally:
            self.model.train(was_training)
        self.count_forward("student_nograd", images.shape[0])
        return torch.flip(probs, dims=[-1])

    def training_step(self, batch, batch_idx):
        batch = self.augment_batch(batch)
        images, targets = batch["image"], batch["segmentation"]
        is_labeled = self.labeled_mask(batch).to(images.device)     # só diagnóstico: a perda não usa GT
        n_centers = images.shape[0]

        if all(key in batch for key in NEIGHBOR_KEYS):
            valid = batch["neighbor_valid"].bool().to(images.device)        # [B,K]
            neighbor_images = batch["neighbors"][valid]                      # [N,C,H,W]
            flow = batch["neighbor_flow"][valid]
            confidence = batch["neighbor_confidence"][valid].to(images)
            center_of_pair = torch.arange(n_centers, device=images.device)[:, None].expand_as(valid)[valid]
        else:
            neighbor_images = images[:0]
            flow = confidence = center_of_pair = None
        n_pairs = neighbor_images.shape[0]

        # Pseudo-rótulo e professor antes do forward com gradiente: o flip põe o aluno em eval por um
        # instante, e é o mesmo aluno (pesos do passo) que o original usa.
        pseudo_label = self.flip_pseudo_label(images)
        with torch.no_grad():
            teacher_logits = self.teacher(images)
        self.count_forward("teacher", n_centers)

        inputs = torch.cat([images, neighbor_images], dim=0)
        if self.use_prototype:
            all_logits, all_local, all_global = self.model.forward_features(inputs)
        else:
            all_logits, all_local, all_global = self.model(inputs), None, None
        self.count_forward("student", inputs.shape[0])
        logits = all_logits[:n_centers]

        seg_loss = self.soft_segmentation_loss(logits, pseudo_label)
        kd_loss = self.kd_loss(logits, teacher_logits)
        zero = logits.sum() * 0.0
        con_loss, proto_loss = zero, zero
        extra: Dict[str, Tensor] = {}

        if n_pairs > 0:
            probs = torch.softmax(all_logits, dim=1)
            current_probs = probs[:n_centers][center_of_pair]
            neighbor_probs = probs[n_centers:]
            transported = warp(neighbor_probs, flow, self.warp_convention)
            con_loss = self.temporal_consistency_loss(current_probs, transported, confidence)
            if self.use_prototype:
                proto_loss = self.prototype_loss(
                    current_probs, neighbor_probs, flow,
                    all_local[:n_centers][center_of_pair], all_local[n_centers:],
                    all_global[:n_centers][center_of_pair], all_global[n_centers:])
            extra["train/noc_mean"] = confidence.mean().detach()
        extra["train/temporal_pairs"] = torch.tensor(float(n_pairs), device=images.device)

        loss = (self.seg_weight * seg_loss + self.kd_weight * kd_loss
                + self.con_weight * con_loss + self.proto_weight * proto_loss)

        # Qualidade (observacional) dos alvos nos frames que TÊM GT: o pseudo-rótulo por flip e o professor.
        if is_labeled.any():
            has_gt = torch.ones(int(is_labeled.sum()), dtype=torch.bool, device=images.device)
            self.log_unlabeled_diagnostics(pseudo_label[is_labeled].argmax(dim=1), targets[is_labeled],
                                           has_gt, prefix="train/flip_pl")
            self.log_unlabeled_diagnostics(teacher_logits[is_labeled].argmax(dim=1), targets[is_labeled],
                                           has_gt, prefix="train/teacher")

        extra.update({
            "train/seg_loss": seg_loss.detach(),
            "train/kd_loss": kd_loss.detach(),
            "train/con_loss": con_loss.detach(),
            "train/proto_loss": proto_loss.detach(),
        })
        # Não há termo supervisionado (sem GT na etapa 2): `sup_loss` = 0 e `unsup_loss` = a perda inteira,
        # com peso 1 -- os pesos de cada termo são fixos e vão nas chaves acima.
        self.log_training(loss, zero.detach(), loss.detach(), 1.0, is_labeled, logits, targets, extra=extra)
        return loss
