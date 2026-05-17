"""Save a side-by-side grid of (raw RGB | saturation channel | post-augmentation)
for a few CAPTCHA samples so you can verify the preprocessing pipeline.

Usage:
    python preview_preprocess.py --out preview.png --n 8
"""
from __future__ import annotations

import argparse
import csv
import os

from PIL import Image
import torch
from torchvision import transforms as T

from src.config import load_config
from src.dataset import SaturationEmphasis, build_train_transform


def denorm(t: torch.Tensor) -> Image.Image:
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    t = (t * std + mean).clamp(0, 1)
    return T.ToPILImage()(t)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--out", default="preview.png")
    parser.add_argument("--n", type=int, default=8)
    args = parser.parse_args()

    cfg = load_config(args.config)
    rows = list(csv.DictReader(open(cfg["data"]["metadata"], encoding="utf-8")))[: args.n]
    sat_only = SaturationEmphasis()
    train_tfm = build_train_transform(cfg)

    cell = 128
    canvas = Image.new("RGB", (cell * 3, cell * args.n), (16, 16, 16))
    for i, row in enumerate(rows):
        raw = Image.open(os.path.join(cfg["data"]["data_dir"], row["filename"])).convert("RGB").resize((cell, cell))
        sat = sat_only(raw).resize((cell, cell))
        aug = denorm(train_tfm(Image.open(os.path.join(cfg["data"]["data_dir"], row["filename"]))))
        canvas.paste(raw, (0, i * cell))
        canvas.paste(sat, (cell, i * cell))
        canvas.paste(aug.resize((cell, cell)), (cell * 2, i * cell))
    canvas.save(args.out)
    print(f"saved {args.out}: rows = (raw | saturation | augmented)")


if __name__ == "__main__":
    main()
