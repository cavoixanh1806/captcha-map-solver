"""Frozen TrOCR encoder + per-position multi-head classifier.

Why this architecture:
- All previous attempts (multi-head CNN from scratch, CRNN+CTC, full TrOCR
  fine-tune with or without space-tokenisation) hit the same wall: 400 train
  samples is not enough to learn shape priors AND alignment AND avoid
  autoregressive exposure bias all at once.
- The encoder of microsoft/trocr-small-printed is a DeiT-small ViT
  pretrained on millions of printed-text images. It already knows what
  letter shapes look like.
- We freeze it (only ~1M trainable params in the head) and train 5 learnable
  query tokens that cross-attend to the encoder feature grid, producing one
  feature vector per character position. Each vector goes through a small
  MLP -> softmax over the 24-symbol alphabet.
- No autoregressive generation, no tokenizer / BPE issues, no exposure bias.
  Per-position cross-entropy is a stable, well-aligned signal.
"""
from __future__ import annotations

import csv
import os
from typing import List

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

from .dataset import (
    ALPHABET,
    CHAR_LEN,
    CLASS_NUM,
    build_or_load_splits,
    lst_to_str,
    str_to_vec,
)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class EncoderClfDataset(Dataset):
    def __init__(
        self,
        data_dir: str,
        metadata_path: str,
        filenames: List[str],
        processor: TrOCRProcessor,
        augment: bool = False,
    ) -> None:
        self.data_dir = data_dir
        self.processor = processor
        meta = {r["filename"]: r["text"] for r in csv.DictReader(open(metadata_path, encoding="utf-8"))}
        self.samples = [(fn, meta[fn]) for fn in filenames]

        if augment:
            from torchvision import transforms as T

            self.aug = T.Compose(
                [
                    T.RandomAffine(degrees=8, translate=(0.05, 0.05), scale=(0.88, 1.12), shear=4, fill=255),
                    T.ColorJitter(brightness=0.20, contrast=0.20, saturation=0.20, hue=0.05),
                ]
            )
        else:
            self.aug = None

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        fname, text = self.samples[idx]
        img = Image.open(os.path.join(self.data_dir, fname)).convert("RGB")
        if self.aug is not None:
            img = self.aug(img)
        pixel_values = self.processor(images=img, return_tensors="pt").pixel_values[0]
        return pixel_values, str_to_vec(text)


# ---------------------------------------------------------------------------
# DataModule
# ---------------------------------------------------------------------------
class EncoderClfDataModule(pl.LightningDataModule):
    def __init__(self, cfg, processor: TrOCRProcessor) -> None:
        super().__init__()
        self.cfg = cfg
        self.processor = processor
        self.train_ds = self.val_ds = self.test_ds = None

    def setup(self, stage: str | None = None) -> None:
        cfg = self.cfg
        split = build_or_load_splits(
            cfg["data"]["metadata"],
            cfg["data"]["splits_file"],
            sizes=(
                cfg["data"]["train_size"],
                cfg["data"]["val_size"],
                cfg["data"]["test_size"],
            ),
            seed=cfg["data"]["seed"],
        )
        self.train_ds = EncoderClfDataset(
            cfg["data"]["data_dir"], cfg["data"]["metadata"], split.train, self.processor, augment=True
        )
        self.val_ds = EncoderClfDataset(
            cfg["data"]["data_dir"], cfg["data"]["metadata"], split.val, self.processor, augment=False
        )
        self.test_ds = EncoderClfDataset(
            cfg["data"]["data_dir"], cfg["data"]["metadata"], split.test, self.processor, augment=False
        )

    def _loader(self, ds, shuffle: bool) -> DataLoader:
        return DataLoader(
            ds,
            batch_size=self.cfg["solver"]["batch_size"],
            shuffle=shuffle,
            num_workers=self.cfg["solver"]["num_workers"],
            pin_memory=True,
            persistent_workers=self.cfg["solver"]["num_workers"] > 0,
        )

    def train_dataloader(self) -> DataLoader:
        return self._loader(self.train_ds, True)

    def val_dataloader(self) -> DataLoader:
        return self._loader(self.val_ds, False)

    def test_dataloader(self) -> DataLoader:
        return self._loader(self.test_ds, False)


# ---------------------------------------------------------------------------
# Lightning model
# ---------------------------------------------------------------------------
class EncoderClf(pl.LightningModule):
    def __init__(self, cfg) -> None:
        super().__init__()
        self.save_hyperparameters({"cfg": cfg})
        self.cfg = cfg

        self.processor = TrOCRProcessor.from_pretrained(cfg["solver"]["pretrained_model"])
        full = VisionEncoderDecoderModel.from_pretrained(cfg["solver"]["pretrained_model"])
        self.encoder = full.encoder

        hidden = self.encoder.config.hidden_size  # 384 for DeiT-small

        # Optionally unfreeze last `n` transformer blocks so the encoder can
        # adapt to captcha-specific glyph styles. Keeping the bulk frozen
        # preserves the pretrained shape prior on 400 samples.
        unfreeze_last = cfg["solver"].get("unfreeze_last_n", 0)
        for p in self.encoder.parameters():
            p.requires_grad = False
        if unfreeze_last > 0:
            blocks = self.encoder.encoder.layer  # DeiTLayer list
            for blk in blocks[-unfreeze_last:]:
                for p in blk.parameters():
                    p.requires_grad = True

        # SPATIAL SPLIT pooling: the encoder grid (24x24 for 384x384 input)
        # is divided into CHAR_LEN equal column-strips. Each strip is
        # average-pooled to a single hidden-dim vector. This provides
        # POSITIONAL DIFFERENTIATION BY CONSTRUCTION, avoiding the mode
        # collapse we saw with learnable queries.
        dropout = cfg["solver"].get("dropout", 0.2)
        self.head = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, CLASS_NUM),
        )
        self._sample_logged = False

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        # Encoder may be partially frozen but we still let autograd handle it
        feat = self.encoder(pixel_values=pixel_values).last_hidden_state
        # Drop the CLS token, reshape patch tokens into a 2D grid
        feat = feat[:, 1:, :]                                # (B, N, D)
        bs, n, d = feat.shape
        h = w = int(n ** 0.5)
        feat = feat.reshape(bs, h, w, d).permute(0, 3, 1, 2)  # (B, D, H, W)
        # Pool to (B, D, 1, CHAR_LEN) -> (B, CHAR_LEN, D)
        feat = F.adaptive_avg_pool2d(feat, (1, CHAR_LEN))
        feat = feat.squeeze(2).transpose(1, 2)               # (B, CHAR_LEN, D)
        return self.head(feat)                               # (B, CHAR_LEN, CLASS_NUM)

    def _step(self, batch):
        x, y = batch
        logits = self(x)
        ls = self.cfg["solver"].get("label_smoothing", 0.0)
        loss = sum(
            F.cross_entropy(logits[:, i, :], y[:, i], label_smoothing=ls)
            for i in range(CHAR_LEN)
        ) / CHAR_LEN
        return loss, logits, y

    @staticmethod
    def _seq_acc(logits, y):
        return ((logits.argmax(-1) == y).all(dim=1)).float().mean()

    @staticmethod
    def _char_acc(logits, y):
        return (logits.argmax(-1) == y).float().mean()

    def training_step(self, batch, batch_idx):
        loss, logits, y = self._step(batch)
        bs = y.size(0)
        self.log("train/loss", loss, prog_bar=True, batch_size=bs)
        self.log("train/seq_acc", self._seq_acc(logits, y), prog_bar=True, batch_size=bs)
        self.log("train/char_acc", self._char_acc(logits, y), batch_size=bs)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, logits, y = self._step(batch)
        bs = y.size(0)
        self.log("val/loss", loss, prog_bar=True, batch_size=bs)
        self.log("val/seq_acc", self._seq_acc(logits, y), prog_bar=True, batch_size=bs)
        self.log("val/char_acc", self._char_acc(logits, y), batch_size=bs)
        return loss

    def test_step(self, batch, batch_idx):
        loss, logits, y = self._step(batch)
        bs = y.size(0)
        preds = logits.argmax(-1)
        self.log("test/loss", loss, batch_size=bs)
        self.log("test/seq_acc", self._seq_acc(logits, y), prog_bar=True, batch_size=bs)
        self.log("test/char_acc", self._char_acc(logits, y), batch_size=bs)
        if not self._sample_logged:
            for i in range(min(10, y.size(0))):
                p = lst_to_str(preds[i])
                t = lst_to_str(y[i])
                print(f"  pred={p} true={t}")
            self._sample_logged = True
        return loss

    def configure_optimizers(self):
        cfg = self.cfg["solver"]
        params = [p for p in self.parameters() if p.requires_grad]
        opt = torch.optim.AdamW(
            params, lr=cfg["lr"], weight_decay=cfg.get("weight_decay", 0.01)
        )
        if cfg.get("scheduler", "none").lower() == "cosine":
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg["epochs"])
            return {"optimizer": opt, "lr_scheduler": sched}
        return opt
