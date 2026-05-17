"""Model definitions following PyCAPTCHA's multi-head architecture.

The classifier is a CNN backbone that produces a single feature vector,
which is reshaped into ``CHAR_LEN`` independent softmax heads of size
``CLASS_NUM``. Cross-entropy is summed over the heads.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tvm
import pytorch_lightning as pl

from .dataset import CHAR_LEN, CLASS_NUM, IDX_TO_CHAR, lst_to_str


# ---------------------------------------------------------------------------
# Backbones
# ---------------------------------------------------------------------------
class ResnetMultiHead(nn.Module):
    def __init__(self, name: str = "resnet18", pretrained: bool = True) -> None:
        super().__init__()
        weights = "DEFAULT" if pretrained else None
        if name == "resnet18":
            backbone = tvm.resnet18(weights=weights)
            in_dim = 512
        elif name == "resnet34":
            backbone = tvm.resnet34(weights=weights)
            in_dim = 512
        else:
            raise ValueError(f"unknown resnet backbone: {name}")
        backbone.fc = nn.Linear(in_dim, CHAR_LEN * CLASS_NUM)
        self.backbone = backbone

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.backbone(x)
        return out.view(out.size(0), CHAR_LEN, CLASS_NUM)


class ConvMultiHead(nn.Module):
    """Light-weight VGG-like CNN matching PyCAPTCHA's `model_conv`.

    Input: 3 x 128 x 128 -> 5 conv blocks with stride-2 pooling -> 4x4 feature
    map flattened into 5 heads of CLASS_NUM logits each.
    """

    def __init__(self, dropout: float = 0.3) -> None:
        super().__init__()

        def block(in_c: int, out_c: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, 3, padding=1),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_c, out_c, 3, padding=1),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2, 2),
            )

        self.features = nn.Sequential(
            block(3, 32),    # 128 -> 64
            block(32, 64),   # 64 -> 32
            block(64, 128),  # 32 -> 16
            block(128, 256), # 16 -> 8
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),  # global pool -> 256
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(256, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, CHAR_LEN * CLASS_NUM),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.classifier(x)
        return x.view(x.size(0), CHAR_LEN, CLASS_NUM)


def build_backbone(cfg) -> nn.Module:
    name = cfg["solver"]["backbone"].lower()
    dropout = cfg["solver"].get("dropout", 0.3)
    if name in {"resnet18", "resnet34"}:
        return ResnetMultiHead(name, pretrained=cfg["solver"]["pretrained"])
    if name == "conv":
        return ConvMultiHead(dropout=dropout)
    raise ValueError(f"unknown backbone: {name}")


# ---------------------------------------------------------------------------
# Lightning module
# ---------------------------------------------------------------------------
def _seq_acc(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """logits: (B, CHAR_LEN, CLASS_NUM); labels: (B, CHAR_LEN)"""
    pred = logits.argmax(dim=-1)
    eq = (pred == labels).all(dim=1).float()
    return eq.mean()


def _char_acc(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    pred = logits.argmax(dim=-1)
    return (pred == labels).float().mean()


class CaptchaModel(pl.LightningModule):
    def __init__(self, cfg) -> None:
        super().__init__()
        self.save_hyperparameters({"cfg": cfg})
        self.cfg = cfg
        self.model = build_backbone(cfg)
        self._sample_logged = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def _step(self, batch):
        x, y = batch
        logits = self(x)
        ls = self.cfg["solver"].get("label_smoothing", 0.0)
        loss = sum(
            F.cross_entropy(logits[:, i, :], y[:, i], label_smoothing=ls)
            for i in range(CHAR_LEN)
        ) / CHAR_LEN
        return loss, logits, y

    def training_step(self, batch, batch_idx):
        loss, logits, y = self._step(batch)
        bs = y.size(0)
        self.log("train/loss", loss, prog_bar=True, batch_size=bs)
        self.log("train/seq_acc", _seq_acc(logits, y), prog_bar=True, batch_size=bs)
        self.log("train/char_acc", _char_acc(logits, y), batch_size=bs)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, logits, y = self._step(batch)
        bs = y.size(0)
        self.log("val/loss", loss, prog_bar=True, batch_size=bs)
        self.log("val/seq_acc", _seq_acc(logits, y), prog_bar=True, batch_size=bs)
        self.log("val/char_acc", _char_acc(logits, y), batch_size=bs)
        return loss

    def test_step(self, batch, batch_idx):
        loss, logits, y = self._step(batch)
        bs = y.size(0)
        self.log("test/loss", loss, batch_size=bs)
        self.log("test/seq_acc", _seq_acc(logits, y), prog_bar=True, batch_size=bs)
        self.log("test/char_acc", _char_acc(logits, y), batch_size=bs)
        if not self._sample_logged:
            preds = logits.argmax(dim=-1)
            for i in range(min(10, preds.size(0))):
                print(f"  pred={lst_to_str(preds[i])}  true={lst_to_str(y[i])}")
            self._sample_logged = True
        return loss

    def configure_optimizers(self):
        cfg = self.cfg["solver"]
        if cfg["optimizer"].lower() == "adamw":
            opt = torch.optim.AdamW(
                self.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"]
            )
        else:
            opt = torch.optim.Adam(self.parameters(), lr=cfg["lr"])

        sched_name = cfg.get("scheduler", "none").lower()
        if sched_name == "cosine":
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg["epochs"])
            return {"optimizer": opt, "lr_scheduler": sched}
        if sched_name == "onecycle":
            steps = max(1, math.ceil(cfg["batch_size"]))  # placeholder, set in trainer
            sched = torch.optim.lr_scheduler.OneCycleLR(
                opt, max_lr=cfg["lr"] * 3, total_steps=cfg["epochs"] * steps
            )
            return {"optimizer": opt, "lr_scheduler": sched}
        return opt
