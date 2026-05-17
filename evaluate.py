"""Evaluate a checkpoint on the held-out test split with confusion analysis."""
from __future__ import annotations

import argparse
from collections import Counter

import torch
from torch.utils.data import DataLoader

from src.config import load_config
from src.dataset import (
    ALPHABET,
    CHAR_LEN,
    CaptchaDataset,
    build_eval_transform,
    build_or_load_splits,
    lst_to_str,
)
from src.model import CaptchaModel


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    args = parser.parse_args()

    cfg = load_config(args.config)
    split = build_or_load_splits(
        cfg["data"]["metadata"],
        cfg["data"]["splits_file"],
        sizes=(cfg["data"]["train_size"], cfg["data"]["val_size"], cfg["data"]["test_size"]),
        seed=cfg["data"]["seed"],
    )
    files = getattr(split, args.split)
    ds = CaptchaDataset(cfg["data"]["data_dir"], cfg["data"]["metadata"], files, build_eval_transform(cfg))
    dl = DataLoader(ds, batch_size=cfg["solver"]["batch_size"], num_workers=0)

    model = CaptchaModel.load_from_checkpoint(args.ckpt)
    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    seq_correct = 0
    char_correct = 0
    total = 0
    confusion: Counter = Counter()
    wrong_examples = []

    with torch.no_grad():
        for x, y in dl:
            x, y = x.to(device), y.to(device)
            preds = model(x).argmax(dim=-1)
            for i in range(y.size(0)):
                t_str = lst_to_str(y[i])
                p_str = lst_to_str(preds[i])
                if p_str == t_str:
                    seq_correct += 1
                else:
                    wrong_examples.append((t_str, p_str))
                for ti, pi in zip(y[i].tolist(), preds[i].tolist()):
                    if ti == pi:
                        char_correct += 1
                    else:
                        confusion[(ALPHABET[ti], ALPHABET[pi])] += 1
                total += 1

    print(f"Split={args.split}  N={total}")
    print(f"Sequence accuracy: {seq_correct/total:.4f}  ({seq_correct}/{total})")
    print(f"Character accuracy: {char_correct/(total*CHAR_LEN):.4f}")
    if wrong_examples:
        print(f"\nWrong examples (showing up to 20):")
        for t, p in wrong_examples[:20]:
            print(f"  true={t}  pred={p}")
    if confusion:
        print(f"\nTop 15 confusing pairs (true -> pred):")
        for (t, p), n in confusion.most_common(15):
            print(f"  {t} -> {p}  : {n}")


if __name__ == "__main__":
    main()
