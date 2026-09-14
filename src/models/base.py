import pytorch_lightning as pl
import torch
from omegaconf import DictConfig, ListConfig, OmegaConf
from torchmetrics.functional.segmentation import mean_iou, dice_score
from ..utils import instantiate_from_config, get_obj_from_str
from ..metrics import absent_class_to_nan, average_symmetric_surface_distance, hausdorff_distance_95

from typing import Dict, Optional, Sequence

# ASSD/HD95 are Python double loops over (batch, class) around `edge_surface_distance`; on
# 8x256x256 batches they cost seconds, versus milliseconds for IoU/Dice. Computing them every
# training step (and every validation epoch) would dominate the wall clock, so by default they are
# only computed where the reported numbers actually come from: the test set.
DEFAULT_SURFACE_METRIC_STAGES = ("test",)


class LitWrapper(pl.LightningModule):
    def __init__(self, model_cfg, optimizer_cfg, lr_scheduler_cfg=None,
                 report_class_ids: Optional[Dict[int, str]] = None,
                 surface_metric_stages: Sequence[str] = DEFAULT_SURFACE_METRIC_STAGES):
        '''
        Args:
            report_class_ids: mapping `{class_index: display_name}` of the foreground classes to
                report individually, e.g. `{1: "C2", 3: "C4"}`. Defaults to every non-background
                class, named `class_<i>`.

                This matters beyond presentation: with `target_variant='multiclass_c2_c4'` the model
                has `n_classes=4` but class 2 (C3) is **never present** in any mask, so averaging
                over all non-background classes silently drags every aggregate down by a third.
                Aggregates here are computed over the reported classes only.
            surface_metric_stages: stages in which ASSD/HD95 are computed (see the note above).
        '''
        super().__init__()
        self.save_hyperparameters(
            {
                "model_cfg": self._to_serializable(model_cfg),
                "optimizer_cfg": self._to_serializable(optimizer_cfg),
                "lr_scheduler_cfg": self._to_serializable(lr_scheduler_cfg),
                "report_class_ids": self._to_serializable(report_class_ids),
                "surface_metric_stages": self._to_serializable(surface_metric_stages),
            }
        )
        self.model_cfg = model_cfg
        self.optimizer_cfg = optimizer_cfg
        self.lr_scheduler_cfg = lr_scheduler_cfg
        self.model = instantiate_from_config(model_cfg)
        self.n_classes = self.model.n_classes
        self.report_class_ids = self._normalize_class_ids(report_class_ids, self.n_classes)
        self.surface_metric_stages = tuple(surface_metric_stages or ())

    @staticmethod
    def _to_serializable(value):
        if isinstance(value, (DictConfig, ListConfig)):
            return OmegaConf.to_container(value, resolve=True)
        if isinstance(value, tuple):
            return list(value)
        return value

    @staticmethod
    def _normalize_class_ids(report_class_ids, n_classes: int) -> Dict[int, str]:
        '''Coerce the config-supplied mapping to `{int: str}` (YAML gives string keys) and validate it.'''
        if report_class_ids is None:
            return {i: f"class_{i}" for i in range(1, n_classes)}

        if isinstance(report_class_ids, DictConfig):
            report_class_ids = OmegaConf.to_container(report_class_ids, resolve=True)

        normalized = {int(class_id): str(name) for class_id, name in dict(report_class_ids).items()}
        for class_id in normalized:
            if not (1 <= class_id < n_classes):
                raise ValueError(
                    f"report_class_ids contains class index {class_id}, which is outside the "
                    f"foreground range [1, {n_classes - 1}] for a model with n_classes={n_classes}. "
                    "Index 0 is background and is never reported."
                )
        return normalized

    def forward(self, x):
        return self.model(x)

    @torch.no_grad()
    def compute_metrics(self, y_logits, y, stage: Optional[str] = None):
        '''
        Per-class and aggregate segmentation metrics.

        Returns a dict with `<metric>/<class name>` for every reported class plus a `<metric>`
        aggregate averaged over those classes only. Tensors from the underlying metrics are `[B, C]`
        with `include_background=False`, so class index `c` lives in column `c - 1`.
        '''
        y_pred = y_logits.argmax(dim=1)
        y_int = y.long()

        per_class = {
            "mean_iou": absent_class_to_nan(mean_iou(
                y_pred, y_int, num_classes=self.n_classes,
                include_background=False, per_class=True, input_format="index",
            )),
            "dice_score": dice_score(
                y_pred, y_int, num_classes=self.n_classes,
                include_background=False, average="none", input_format="index",
            ),
        }

        if stage is None or stage in self.surface_metric_stages:
            per_class["assd"] = average_symmetric_surface_distance(
                y_pred, y_int, num_classes=self.n_classes, include_background=False
            )
            per_class["hd95"] = hausdorff_distance_95(
                y_pred, y_int, num_classes=self.n_classes, include_background=False
            )

        metrics = {}
        for metric_name, values in per_class.items():
            class_means = []
            for class_id, class_name in self.report_class_ids.items():
                # `.nanmean()` is required: ASSD/HD95 are NaN for a class present in only one of
                # prediction/target (see src/metrics.py).
                value = values[:, class_id - 1].nanmean()
                metrics[f"{metric_name}/{class_name}"] = value
                class_means.append(value)
            metrics[metric_name] = torch.stack(class_means).nanmean()

        return metrics

    def compute_loss(self, y_logits, y):
        y = y.long()
        return torch.nn.functional.cross_entropy(y_logits, y)

    def calculate_loss_and_metrics(self, y_logits, y, stage):
        loss = self.compute_loss(y_logits, y)
        metrics_dict = self.compute_metrics(y_logits, y, stage=stage)

        # Create a dictionary to log losses and metrics
        loss_dict = {f"{stage}/loss": loss}
        for metric_name, metric_value in metrics_dict.items():
            loss_dict[f"{stage}/{metric_name}"] = metric_value

        return loss, loss_dict

    def shared_step(self, batch, batch_idx, stage):
        x, y = batch["image"], batch["segmentation"]
        y_pred = self.forward(x)
        loss, loss_dict = self.calculate_loss_and_metrics(y_pred, y, stage)

        self.log(
            "step",
            self.global_step,
            prog_bar=True,
            logger=True,
            on_step=True,
            on_epoch=False,
        )
        self.log_dict(
            loss_dict, prog_bar=True, logger=True, on_step=False, on_epoch=True
        )

        return loss, loss_dict, y_pred, y

    def training_step(self, batch, batch_idx):
        loss, *_ = self.shared_step(batch, batch_idx, stage="train")
        return loss

    def validation_step(self, batch, batch_idx):
        loss, *_ = self.shared_step(batch, batch_idx, stage="val")
        return loss

    def test_step(self, batch, batch_idx):
        loss, *_ = self.shared_step(batch, batch_idx, stage="test")
        return loss

    def predict_step(self, batch, batch_idx):
        *_, y_pred, y = self.shared_step(batch, batch_idx, stage="predict")
        return y_pred, y

    def optimizer_param_groups(self):
        '''
        The parameters handed to the optimizer.

        Load-bearing by omission: `MeanTeacherLitWrapper.teacher` is an EMA copy, not a trained
        module, and it stays out of the optimizer precisely because this returns
        `self.model.parameters()` and nothing else (pinned by
        `tests/test_ssl_methods.py::TestMeanTeacher.test_teacher_is_not_in_the_optimizer`).

        A subclass that owns *additional trainable* modules -- `DiffRectLitWrapper.rectifier` --
        must override this. Forgetting to is a bug that runs perfectly: the module is built,
        forwarded, checkpointed, and never updated.
        '''
        return self.model.parameters()

    def configure_optimizers(self):
        optimizer_cls = get_obj_from_str(self.optimizer_cfg["target"])
        optimizer_params = self.optimizer_cfg.get("params", {})
        optimizer = optimizer_cls(
            self.optimizer_param_groups(), **optimizer_params
        )
        if self.lr_scheduler_cfg is None:
            return optimizer

        scheduler_cls = get_obj_from_str(self.lr_scheduler_cfg["target"])
        scheduler_params = self.lr_scheduler_cfg.get("params", {})
        scheduler = scheduler_cls(optimizer, **scheduler_params)

        scheduler_config = {"scheduler": scheduler}
        for key in ("interval", "frequency", "monitor", "strict", "name"):
            value = self.lr_scheduler_cfg.get(key)
            if value is not None:
                scheduler_config[key] = value

        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler_config,
        }
