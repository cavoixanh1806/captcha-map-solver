"""Dataset / DataModule for the map_*.png CAPTCHA corpus.

The dataset has 500 samples of 128x128 RGBA images with 5-character labels
drawn from the 24-symbol alphabet documented in ``configs/default.yaml``.

The PyCAPTCHA-style label representation is a 1-D LongTensor of length
``CHAR_LEN``, with each entry mapped via ``CHAR_TO_IDX``.
"""
from __future__ import annotations

import csv
import json
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T

import pytorch_lightning as pl


# ---------------------------------------------------------------------------
# Alphabet
# ---------------------------------------------------------------------------
ALPHABET = "3479ACDEFHJKLMNPQRTUVWXY"
CHAR_LEN = 5
CLASS_NUM = len(ALPHABET)
CHAR_TO_IDX = {c: i for i, c in enumerate(ALPHABET)}
IDX_TO_CHAR = {i: c for i, c in enumerate(ALPHABET)}


def str_to_vec(text: str) -> torch.Tensor:
    """Encode a CAPTCHA string of length CHAR_LEN to a LongTensor."""
    if len(text) != CHAR_LEN:
        raise ValueError(f"label '{text}' must have length {CHAR_LEN}")
    return torch.tensor([CHAR_TO_IDX[c] for c in text], dtype=torch.long)


def lst_to_str(indices) -> str:
    """Decode a 1-D iterable of indices back to a string."""
    return "".join(IDX_TO_CHAR[int(i)] for i in indices)


# ---------------------------------------------------------------------------
# Train / val / test split helper
# ---------------------------------------------------------------------------
@dataclass
class Split:
    train: List[str]
    val: List[str]
    test: List[str]


def build_or_load_splits(
    metadata_path: str,
    splits_file: str,
    sizes: Tuple[int, int, int] = (400, 50, 50),
    seed: int = 42,
) -> Split:
    """Create a deterministic split file or reuse an existing one."""
    if os.path.exists(splits_file):
        with open(splits_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        return Split(train=data["train"], val=data["val"], test=data["test"])

    rows = list(csv.DictReader(open(metadata_path, encoding="utf-8")))
    filenames = [r["filename"] for r in rows]
    if sum(sizes) != len(filenames):
        raise ValueError(
            f"split sizes {sizes} do not match dataset size {len(filenames)}"
        )

    rng = random.Random(seed)
    shuffled = filenames[:]
    rng.shuffle(shuffled)
    train_n, val_n, test_n = sizes
    split = Split(
        train=shuffled[:train_n],
        val=shuffled[train_n : train_n + val_n],
        test=shuffled[train_n + val_n : train_n + val_n + test_n],
    )

    Path(splits_file).parent.mkdir(parents=True, exist_ok=True)
    with open(splits_file, "w", encoding="utf-8") as f:
        json.dump({"train": split.train, "val": split.val, "test": split.test}, f, indent=2)
    return split


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class CaptchaDataset(Dataset):
    def __init__(
        self,
        data_dir: str,
        metadata_path: str,
        filenames: List[str],
        transform=None,
    ) -> None:
        self.data_dir = data_dir
        self.transform = transform
        meta = {r["filename"]: r["text"] for r in csv.DictReader(open(metadata_path, encoding="utf-8"))}
        self.samples = [(fn, meta[fn]) for fn in filenames]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        fname, text = self.samples[idx]
        img = Image.open(os.path.join(self.data_dir, fname)).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, str_to_vec(text)


# ---------------------------------------------------------------------------
# Transforms
# ---------------------------------------------------------------------------
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class SaturationEmphasis:
    """Replace each RGB image with a 3-channel saturation map.

    Why: this dataset randomises the colour of every glyph, so a model trained
    on raw RGB tends to memorise (pixel, hue) -> char mappings on the 400 train
    samples and fail on validation. The HSV saturation channel separates the
    pastel background (low S) from the colourful glyphs (high S) regardless of
    the specific glyph hue, forcing the network to learn shape features.
    """

    def __call__(self, img: Image.Image) -> Image.Image:
        if img.mode != "RGB":
            img = img.convert("RGB")
        _, s, _ = img.convert("HSV").split()
        return Image.merge("RGB", (s, s, s))


def build_train_transform(cfg) -> T.Compose:
    a = cfg["augment"]
    h, w = cfg["image"]["height"], cfg["image"]["width"]
    return T.Compose(
        [
            SaturationEmphasis(),
            T.Resize((h, w)),
            T.RandomAffine(
                degrees=a["rotation_degrees"],
                translate=(a["translate"], a["translate"]),
                scale=(a["scale_min"], a["scale_max"]),
                shear=a["shear"],
                fill=0,
            ),
            T.ColorJitter(
                brightness=a["brightness"],
                contrast=a["contrast"],
                saturation=0,
                hue=0,
            ),
            T.ToTensor(),
            T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            T.RandomErasing(
                p=a["random_erasing_prob"],
                scale=(a["random_erasing_scale_min"], a["random_erasing_scale_max"]),
                value=0,
            ),
        ]
    )


def build_eval_transform(cfg) -> T.Compose:
    h, w = cfg["image"]["height"], cfg["image"]["width"]
    return T.Compose(
        [
            SaturationEmphasis(),
            T.Resize((h, w)),
            T.ToTensor(),
            T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


# ---------------------------------------------------------------------------
# DataModule
# ---------------------------------------------------------------------------
class CaptchaDataModule(pl.LightningDataModule):
    def __init__(self, cfg) -> None:
        super().__init__()
        self.cfg = cfg
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
        train_t = build_train_transform(cfg)
        eval_t = build_eval_transform(cfg)
        self.train_ds = CaptchaDataset(cfg["data"]["data_dir"], cfg["data"]["metadata"], split.train, train_t)
        self.val_ds = CaptchaDataset(cfg["data"]["data_dir"], cfg["data"]["metadata"], split.val, eval_t)
        self.test_ds = CaptchaDataset(cfg["data"]["data_dir"], cfg["data"]["metadata"], split.test, eval_t)

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
        return self._loader(self.train_ds, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        return self._loader(self.val_ds, shuffle=False)

    def test_dataloader(self) -> DataLoader:
        return self._loader(self.test_ds, shuffle=False)
