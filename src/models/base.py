import pytorch_lightning as pl
import torch
from omegaconf import DictConfig, OmegaConf
from torchmetrics.functional.segmentation import mean_iou, dice_score
from ..utils import instantiate_from_config, get_obj_from_str


class LitWrapper(pl.LightningModule):
    def __init__(self, model_cfg, optimizer_cfg):
        super().__init__()
        self.save_hyperparameters(
            {
                "model_cfg": self._to_serializable(model_cfg),
                "optimizer_cfg": self._to_serializable(optimizer_cfg),
            }
        )
        self.model_cfg = model_cfg
        self.optimizer_cfg = optimizer_cfg
        self.model = instantiate_from_config(model_cfg)
        self.n_classes = self.model.n_classes

    @staticmethod
    def _to_serializable(value):
        if isinstance(value, DictConfig):
            return OmegaConf.to_container(value, resolve=True)
        return value

    def forward(self, x):
        return self.model(x)

    @torch.no_grad()
    def compute_metrics(self, y_logits, y):
        y_pred = y_logits.argmax(dim=1)
        y_int = y.int()

        metrics_dict = dict()
        mean_iou_score = mean_iou(
            y_pred, y_int, num_classes=self.model.n_classes, include_background=False
        ).mean()

        dice = dice_score(
            y_pred,
            y_int,
            num_classes=self.model.n_classes,
            include_background=False,
            average="micro",
        ).mean()

        metrics_dict = {
            "mean_iou": mean_iou_score,
            "dice_score": dice,
        }

        return metrics_dict

    def compute_loss(self, y_logits, y):
        # y_pred = y_logits.squeeze(1)
        y = y.long()
        return torch.nn.functional.cross_entropy(y_logits, y)

    def calculate_loss_and_metrics(self, y_logits, y, stage):
        loss = self.compute_loss(y_logits, y)
        metrics_dict = self.compute_metrics(y_logits, y)

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

    def configure_optimizers(self):
        optimizer_cls = get_obj_from_str(self.optimizer_cfg["target"])
        optimizer_params = self.optimizer_cfg.get("params", {})
        optimizer = optimizer_cls(
            self.model.parameters(), **optimizer_params
        )
        return optimizer
