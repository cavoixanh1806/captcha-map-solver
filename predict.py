"""Single-image inference for either the multi-head, CTC, or TrOCR checkpoint.

Examples:
    python predict.py --ckpt checkpoints/baseline-resnet18/last.ckpt --image data/map_00000.png
    python predict.py --ckpt checkpoints/crnn-ctc/last.ckpt --image data/map_00000.png --task ctc
    python predict.py --ckpt checkpoints/trocr-small/last.ckpt --image data/map_00000.png --task trocr --config configs/trocr.yaml
"""
from __future__ import annotations

import argparse

import torch
from PIL import Image

from src.config import load_config
from src.dataset import ALPHABET, build_eval_transform, lst_to_str


def predict(ckpt: str, image: str, config: str, task: str) -> str:
    cfg = load_config(config)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if task == "trocr":
        from src.trocr import TrOCRLitModel

        model = TrOCRLitModel.load_from_checkpoint(ckpt)
        model.eval().to(device)
        img = Image.open(image).convert("RGB")
        pixel_values = model.processor(images=img, return_tensors="pt").pixel_values.to(device)
        with torch.no_grad():
            gen = model.model.generate(
                pixel_values, max_length=cfg["solver"]["max_length"], num_beams=cfg["solver"]["num_beams"]
            )
        return model.processor.batch_decode(gen, skip_special_tokens=True)[0].replace(" ", "").upper()

    if task == "ctc":
        from src.crnn import CTCCaptchaModel, _ctc_decode

        model = CTCCaptchaModel.load_from_checkpoint(ckpt)
        model.eval().to(device)
        tfm = build_eval_transform(cfg)
        img = tfm(Image.open(image).convert("RGB")).unsqueeze(0).to(device)
        with torch.no_grad():
            logits = model(img)
        decoded = _ctc_decode(logits)[0]
        return "".join(ALPHABET[c] for c in decoded)

    from src.model import CaptchaModel

    model = CaptchaModel.load_from_checkpoint(ckpt)
    model.eval().to(device)
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
    parser.add_argument("--task", default="head", choices=["head", "ctc", "trocr"])
    args = parser.parse_args()
    text = predict(args.ckpt, args.image, args.config, args.task)
    print(text)


if __name__ == "__main__":
    main()
