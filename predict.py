"""Single-image inference.

Example:
    python predict.py --ckpt checkpoints/baseline-resnet18/last.ckpt --image data/map_00000.png
"""
from __future__ import annotations

import argparse

import torch
from PIL import Image

from src.config import load_config
from src.dataset import build_eval_transform, lst_to_str
from src.model import CaptchaModel


def predict(ckpt: str, image: str, config: str) -> str:
    cfg = load_config(config)
    model = CaptchaModel.load_from_checkpoint(ckpt)
    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    tfm = build_eval_transform(cfg)
    img = tfm(Image.open(image).convert("RGB")).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(img)
    pred = logits.argmax(dim=-1).squeeze(0).cpu().tolist()
    return lst_to_str(pred)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()
    text = predict(args.ckpt, args.image, args.config)
    print(text)


if __name__ == "__main__":
    main()
