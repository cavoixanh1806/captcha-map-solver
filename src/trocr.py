"""TrOCR fine-tuning module for the map_*.png CAPTCHA dataset.

This is the strategically-chosen tier-3 path from research.md. We fine-tune
``microsoft/trocr-small-printed`` (61M params) so the encoder's pretrained
ViT/DeiT features bring the prior knowledge that the 400-sample training set
cannot supply. The decoder is a small autoregressive Transformer; we leave it
intact and rely on the optimizer to bend it toward the 24-symbol alphabet.

The dataset uses raw RGBA -> RGB conversion (no saturation trick): TrOCR has
seen colourful real-world images and the saturation hack would actively hurt
because it removes information the encoder already knows how to use.
"""
from __future__ import annotations

from typing import List

import pytorch_lightning as pl
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from transformers import (
    TrOCRProcessor,
    VisionEncoderDecoderModel,
    get_cosine_schedule_with_warmup,
)
import csv
import os

from .dataset import ALPHABET, CHAR_LEN, build_or_load_splits


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class TrOCRDataset(Dataset):
    """Raw RGB images + plain-text labels; processor handles tensor conversion."""

    def __init__(
        self,
        data_dir: str,
        metadata_path: str,
        filenames: List[str],
        processor: TrOCRProcessor,
        max_length: int,
        augment: bool = False,
    ) -> None:
        self.data_dir = data_dir
        self.processor = processor
        self.max_length = max_length
        self.augment = augment
        meta = {r["filename"]: r["text"] for r in csv.DictReader(open(metadata_path, encoding="utf-8"))}
        self.samples = [(fn, meta[fn]) for fn in filenames]

        if augment:
            from torchvision import transforms as T

            self.aug = T.Compose(
                [
                    T.RandomAffine(degrees=6, translate=(0.04, 0.04), scale=(0.92, 1.08), shear=3, fill=255),
                    T.ColorJitter(brightness=0.15, contrast=0.15),
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
        # CRITICAL: BPE merges adjacent letters (e.g. "TN" -> single token).
        # Inserting spaces forces every captcha char to tokenise to a single
        # "_<char>" token, giving a fixed-length token sequence the decoder
        # can learn reliably.
        spaced = " ".join(text)
        labels = self.processor.tokenizer(
            spaced,
            padding="max_length",
            max_length=self.max_length,
            truncation=True,
            return_tensors="pt",
        ).input_ids[0]
        # HF convention: label PAD ids set to -100 so they are ignored in loss
        labels[labels == self.processor.tokenizer.pad_token_id] = -100
        return {"pixel_values": pixel_values, "labels": labels, "text": text}


# ---------------------------------------------------------------------------
# DataModule
# ---------------------------------------------------------------------------
class TrOCRDataModule(pl.LightningDataModule):
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
        max_length = cfg["solver"]["max_length"]
        self.train_ds = TrOCRDataset(
            cfg["data"]["data_dir"], cfg["data"]["metadata"], split.train, self.processor, max_length, augment=True
        )
        self.val_ds = TrOCRDataset(
            cfg["data"]["data_dir"], cfg["data"]["metadata"], split.val, self.processor, max_length, augment=False
        )
        self.test_ds = TrOCRDataset(
            cfg["data"]["data_dir"], cfg["data"]["metadata"], split.test, self.processor, max_length, augment=False
        )

    def _loader(self, ds, shuffle: bool) -> DataLoader:
        return DataLoader(
            ds,
            batch_size=self.cfg["solver"]["batch_size"],
            shuffle=shuffle,
            num_workers=self.cfg["solver"]["num_workers"],
            pin_memory=True,
            persistent_workers=self.cfg["solver"]["num_workers"] > 0,
            collate_fn=self._collate,
        )

    @staticmethod
    def _collate(batch):
        pixel_values = torch.stack([b["pixel_values"] for b in batch])
        labels = torch.stack([b["labels"] for b in batch])
        texts = [b["text"] for b in batch]
        return {"pixel_values": pixel_values, "labels": labels, "texts": texts}

    def train_dataloader(self) -> DataLoader:
        return self._loader(self.train_ds, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        return self._loader(self.val_ds, shuffle=False)

    def test_dataloader(self) -> DataLoader:
        return self._loader(self.test_ds, shuffle=False)


# ---------------------------------------------------------------------------
# Lightning module
# ---------------------------------------------------------------------------
class TrOCRLitModel(pl.LightningModule):
    def __init__(self, cfg) -> None:
        super().__init__()
        self.save_hyperparameters({"cfg": cfg})
        self.cfg = cfg

        self.processor = TrOCRProcessor.from_pretrained(cfg["solver"]["pretrained_model"])
        self.model = VisionEncoderDecoderModel.from_pretrained(cfg["solver"]["pretrained_model"])

        # Token ids that the encoder-decoder needs at construction time
        self.model.config.decoder_start_token_id = self.processor.tokenizer.cls_token_id
        self.model.config.pad_token_id = self.processor.tokenizer.pad_token_id
        self.model.config.eos_token_id = self.processor.tokenizer.sep_token_id
        self.model.config.vocab_size = self.model.config.decoder.vocab_size

        # transformers >= 4.40 requires generation defaults on generation_config
        gc = self.model.generation_config
        gc.decoder_start_token_id = self.processor.tokenizer.cls_token_id
        gc.pad_token_id = self.processor.tokenizer.pad_token_id
        gc.eos_token_id = self.processor.tokenizer.sep_token_id
        gc.max_length = cfg["solver"]["max_length"]
        gc.num_beams = cfg["solver"]["num_beams"]
        gc.early_stopping = True
        gc.no_repeat_ngram_size = 0
        gc.length_penalty = 1.0
        gc.repetition_penalty = cfg["solver"].get("repetition_penalty", 1.0)
        # We DO NOT set bos_token_id: TrOCR uses decoder_start_token_id only.

        # Precompute alphabet token IDs for constrained decoding (optional).
        # WARNING: a BPE tokenizer can merge adjacent characters into single
        # pieces (e.g. "4TCYW" might tokenise to ["4T", "CY", "W"]). In that
        # case forcing exactly CHAR_LEN single-char tokens during generation
        # is inconsistent with the training distribution and HURTS accuracy.
        # We keep the helper so the user can experiment, but the default
        # config keeps `constrained_decoding: false`.
        self._alphabet_token_ids = self._build_alphabet_token_ids()
        self._sample_logged = False

    def _build_alphabet_token_ids(self) -> List[int] | None:
        tok = self.processor.tokenizer
        ids: List[int] = []
        for ch in ALPHABET:
            piece = tok(ch, add_special_tokens=False).input_ids
            if len(piece) != 1:
                return None
            ids.append(piece[0])
        return ids

    def _prefix_allowed_tokens_fn(self, batch_id: int, input_ids):
        # input_ids is a 1-D LongTensor of tokens already generated for this
        # beam, including the decoder_start token at position 0.
        # With space-tokenised labels, each captcha char becomes one "_<char>"
        # token, so positions 1..CHAR_LEN must be alphabet, position
        # CHAR_LEN+1 must be EOS, then PAD.
        pos = int(input_ids.shape[-1])  # number of tokens emitted so far
        if pos <= CHAR_LEN:
            return self._alphabet_token_ids
        if pos == CHAR_LEN + 1:
            return [self.processor.tokenizer.sep_token_id]
        return [self.processor.tokenizer.pad_token_id]

    # ------------------------------------------------------------------
    # Steps
    # ------------------------------------------------------------------
    def training_step(self, batch, batch_idx):
        out = self.model(pixel_values=batch["pixel_values"], labels=batch["labels"])
        bs = batch["pixel_values"].size(0)
        self.log("train/loss", out.loss, prog_bar=True, batch_size=bs)
        return out.loss

    def _generate_decode(self, pixel_values: torch.Tensor) -> List[str]:
        kwargs = {}
        if self._alphabet_token_ids is not None and self.cfg["solver"].get(
            "constrained_decoding", False
        ):
            kwargs["prefix_allowed_tokens_fn"] = self._prefix_allowed_tokens_fn
        gen = self.model.generate(pixel_values, **kwargs)
        decoded = self.processor.batch_decode(gen, skip_special_tokens=True)
        # Labels were trained with spaces between chars to force per-char
        # tokenisation; strip them back out and uppercase for comparison.
        return [d.replace(" ", "").upper() for d in decoded]

    def _eval_step(self, batch, prefix: str) -> torch.Tensor:
        out = self.model(pixel_values=batch["pixel_values"], labels=batch["labels"])
        bs = batch["pixel_values"].size(0)
        self.log(f"{prefix}/loss", out.loss, prog_bar=True, batch_size=bs)

        preds = self._generate_decode(batch["pixel_values"])
        truths = batch["texts"]
        seq_correct = 0
        char_correct = 0
        char_total = 0
        for p, t in zip(preds, truths):
            char_total += len(t)
            for i in range(len(t)):
                if i < len(p) and p[i] == t[i]:
                    char_correct += 1
            if p == t:
                seq_correct += 1
        self.log(f"{prefix}/seq_acc", seq_correct / max(1, len(truths)), prog_bar=True, batch_size=bs)
        self.log(f"{prefix}/char_acc", char_correct / max(1, char_total), batch_size=bs)

        if prefix == "test" and not self._sample_logged:
            for i in range(min(10, len(preds))):
                print(f"  pred={preds[i]:<8s} true={truths[i]}")
            self._sample_logged = True
        return out.loss

    def validation_step(self, batch, batch_idx):
        return self._eval_step(batch, "val")

    def test_step(self, batch, batch_idx):
        return self._eval_step(batch, "test")

    # ------------------------------------------------------------------
    # Optim
    # ------------------------------------------------------------------
    def configure_optimizers(self):
        cfg = self.cfg["solver"]
        opt = torch.optim.AdamW(
            self.parameters(), lr=cfg["lr"], weight_decay=cfg.get("weight_decay", 0.0)
        )
        # Estimate total steps for the cosine schedule
        steps_per_epoch = max(
            1,
            self.cfg["data"]["train_size"]
            // (cfg["batch_size"] * cfg.get("grad_accumulation", 1)),
        )
        total_steps = steps_per_epoch * cfg["epochs"]
        sched = get_cosine_schedule_with_warmup(
            opt,
            num_warmup_steps=cfg.get("warmup_steps", 0),
            num_training_steps=total_steps,
        )
        return {
            "optimizer": opt,
            "lr_scheduler": {"scheduler": sched, "interval": "step"},
        }
