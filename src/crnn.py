"""CRNN + CTC implementation for the map_*.png CAPTCHA dataset.

Inspired by the Keras OCR-for-CAPTCHA example
(https://keras.io/examples/vision/captcha_ocr/) but reimplemented in
PyTorch / Lightning so that it can share the existing project pipeline.

Why CRNN + CTC for this dataset:
- Multi-head CNN baselines (ResNet-18 / ConvNet) overfit hard on the 400-sample
  training split. The model memorises position + colour patterns and val acc
  collapses to 0.
- Saturation-channel preprocessing removes the random per-glyph colour signal,
  but a 11M parameter ResNet still memorises shape + line patterns.
- CRNN is smaller (~3M params with 256-wide LSTM) and the CTC loss enforces
  *sequential* learning: the model has to localise glyphs along the width axis
  rather than memorise the whole image, which fits this dataset much better.

Reference architecture: VGG-style conv stack -> width-preserving pooling -> LSTM
-> linear -> log-softmax over (alphabet + blank).
"""
from __future__ import annotations

from typing import List

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F

from .dataset import ALPHABET, CHAR_LEN


# ---------------------------------------------------------------------------
# Vocab (CTC reserves index 0 for the blank token)
# ---------------------------------------------------------------------------
CTC_BLANK = 0
CTC_NUM_CLASSES = len(ALPHABET) + 1  # 24 chars + blank = 25


def _ctc_decode(logits: torch.Tensor) -> List[List[int]]:
    """Greedy CTC decoder. logits shape: (T, B, C) -> list of int sequences."""
    pred = logits.argmax(dim=-1).transpose(0, 1)  # (B, T)
    out: List[List[int]] = []
    for row in pred.tolist():
        merged: List[int] = []
        prev = -1
        for idx in row:
            if idx != prev and idx != CTC_BLANK:
                merged.append(idx - 1)  # shift back to 0..23 alphabet space
            prev = idx
        out.append(merged)
    return out


# ---------------------------------------------------------------------------
# CRNN backbone
# ---------------------------------------------------------------------------
class CRNN(nn.Module):
    """3 x 128 x 128 -> (T=31) x B x num_classes."""

    def __init__(self, in_channels: int = 3, hidden: int = 256, num_classes: int = CTC_NUM_CLASSES) -> None:
        super().__init__()

        def conv_bn(c_in: int, c_out: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Conv2d(c_in, c_out, 3, padding=1),
                nn.BatchNorm2d(c_out),
                nn.ReLU(inplace=True),
            )

        self.cnn = nn.Sequential(
            conv_bn(in_channels, 32),
            nn.MaxPool2d(2, 2),                 # 32 x 64 x 64
            conv_bn(32, 64),
            nn.MaxPool2d(2, 2),                 # 64 x 32 x 32
            conv_bn(64, 128),
            conv_bn(128, 128),
            nn.MaxPool2d((2, 1), (2, 1)),       # 128 x 16 x 32
            conv_bn(128, 256),
            conv_bn(256, 256),
            nn.MaxPool2d((2, 1), (2, 1)),       # 256 x 8 x 32
            conv_bn(256, 256),
            nn.MaxPool2d((2, 1), (2, 1)),       # 256 x 4 x 32
            conv_bn(256, 256),
            nn.MaxPool2d((2, 1), (2, 1)),       # 256 x 2 x 32
            nn.Conv2d(256, 256, (2, 2), stride=1, padding=0),
            nn.ReLU(inplace=True),              # 256 x 1 x 31
        )

        self.dropout = nn.Dropout(0.3)
        self.rnn = nn.LSTM(
            input_size=256,
            hidden_size=hidden,
            num_layers=2,
            bidirectional=True,
            dropout=0.3,
        )
        self.fc = nn.Linear(hidden * 2, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.cnn(x)               # B x 256 x 1 x T
        feat = feat.squeeze(2)           # B x 256 x T
        feat = feat.permute(2, 0, 1)     # T x B x 256
        feat = self.dropout(feat)
        rnn_out, _ = self.rnn(feat)      # T x B x 2H
        return self.fc(rnn_out)          # T x B x C


# ---------------------------------------------------------------------------
# Lightning module
# ---------------------------------------------------------------------------
class CTCCaptchaModel(pl.LightningModule):
    def __init__(self, cfg) -> None:
        super().__init__()
        self.save_hyperparameters({"cfg": cfg})
        self.cfg = cfg
        self.model = CRNN(
            in_channels=3,
            hidden=cfg["solver"].get("rnn_hidden", 256),
            num_classes=CTC_NUM_CLASSES,
        )
        self.ctc = nn.CTCLoss(blank=CTC_BLANK, zero_infinity=True)
        self._sample_logged = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    # ----- shared step -----
    def _step(self, batch):
        x, y = batch                              # y in 0..23
        logits = self.model(x)                    # T x B x C
        log_probs = F.log_softmax(logits, dim=-1)
        t_size = log_probs.size(0)
        b_size = log_probs.size(1)
        # CTC labels live in 1..24 (blank=0)
        targets = (y + 1).long().flatten()
        input_lengths = torch.full((b_size,), t_size, dtype=torch.long, device=logits.device)
        target_lengths = torch.full((b_size,), CHAR_LEN, dtype=torch.long, device=logits.device)
        loss = self.ctc(log_probs, targets, input_lengths, target_lengths)
        return loss, logits, y

    # ----- metric helpers -----
    @staticmethod
    def _seq_char_accuracy(logits: torch.Tensor, labels: torch.Tensor):
        decoded = _ctc_decode(logits)
        truth = labels.tolist()
        seq_correct = 0
        char_correct = 0
        char_total = 0
        for pred_seq, true_seq in zip(decoded, truth):
            char_total += len(true_seq)
            # compare with truncation/padding to fixed CHAR_LEN
            for i in range(len(true_seq)):
                if i < len(pred_seq) and pred_seq[i] == true_seq[i]:
                    char_correct += 1
            if pred_seq == true_seq:
                seq_correct += 1
        return (
            seq_correct / max(1, len(truth)),
            char_correct / max(1, char_total),
        )

    # ----- lightning hooks -----
    def training_step(self, batch, batch_idx):
        loss, logits, y = self._step(batch)
        seq_acc, char_acc = self._seq_char_accuracy(logits.detach(), y)
        bs = y.size(0)
        self.log("train/loss", loss, prog_bar=True, batch_size=bs)
        self.log("train/seq_acc", seq_acc, prog_bar=True, batch_size=bs)
        self.log("train/char_acc", char_acc, batch_size=bs)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, logits, y = self._step(batch)
        seq_acc, char_acc = self._seq_char_accuracy(logits, y)
        bs = y.size(0)
        self.log("val/loss", loss, prog_bar=True, batch_size=bs)
        self.log("val/seq_acc", seq_acc, prog_bar=True, batch_size=bs)
        self.log("val/char_acc", char_acc, batch_size=bs)
        return loss

    def test_step(self, batch, batch_idx):
        loss, logits, y = self._step(batch)
        seq_acc, char_acc = self._seq_char_accuracy(logits, y)
        bs = y.size(0)
        self.log("test/loss", loss, batch_size=bs)
        self.log("test/seq_acc", seq_acc, prog_bar=True, batch_size=bs)
        self.log("test/char_acc", char_acc, batch_size=bs)
        if not self._sample_logged:
            decoded = _ctc_decode(logits)
            for i in range(min(10, y.size(0))):
                p = "".join(ALPHABET[c] for c in decoded[i])
                t = "".join(ALPHABET[c] for c in y[i].tolist())
                print(f"  pred={p:<8s} true={t}")
            self._sample_logged = True
        return loss

    def configure_optimizers(self):
        cfg = self.cfg["solver"]
        opt_cls = torch.optim.AdamW if cfg["optimizer"].lower() == "adamw" else torch.optim.Adam
        opt = opt_cls(self.parameters(), lr=cfg["lr"], weight_decay=cfg.get("weight_decay", 0.0))

        sched_name = cfg.get("scheduler", "none").lower()
        if sched_name == "cosine":
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg["epochs"])
            return {"optimizer": opt, "lr_scheduler": sched}
        return opt
