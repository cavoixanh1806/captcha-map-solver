"""Verify how TrOCR's tokenizer splits captcha labels.

Run on the train machine (after `pip install transformers sentencepiece`).
Output tells us whether each captcha char becomes one BPE token or multiple,
which determines whether constrained per-position decoding is even possible.
"""
from __future__ import annotations

import csv
from collections import Counter
from transformers import TrOCRProcessor


MODEL = "microsoft/trocr-small-printed"
ALPHABET = "3479ACDEFHJKLMNPQRTUVWXY"


def main() -> None:
    proc = TrOCRProcessor.from_pretrained(MODEL)
    tok = proc.tokenizer

    print("=" * 60)
    print("Per-char tokenization (in isolation):")
    print("=" * 60)
    bad_chars = []
    for ch in ALPHABET:
        ids = tok(ch, add_special_tokens=False).input_ids
        pieces = tok.convert_ids_to_tokens(ids)
        marker = "OK" if len(ids) == 1 else "MULTI"
        print(f"  {ch}: {ids} {pieces}  [{marker}]")
        if len(ids) != 1:
            bad_chars.append(ch)

    rows = list(csv.DictReader(open("data/metadata.csv", encoding="utf-8")))
    sample_labels = [r["text"] for r in rows[:10]]

    print("\n" + "=" * 60)
    print("Full-label tokenization (with special tokens):")
    print("=" * 60)
    length_dist = Counter()
    for lbl in sample_labels:
        ids = tok(lbl, return_tensors=None).input_ids
        pieces = tok.convert_ids_to_tokens(ids)
        decoded = tok.decode(ids, skip_special_tokens=True)
        print(f"  '{lbl}' -> {len(ids)} tokens: {pieces}  ->  decode='{decoded}'")
        length_dist[len(ids)] += 1

    print("\nLength distribution across 10 samples:", dict(length_dist))
    print(f"\nMulti-token chars: {bad_chars or 'NONE'}")

    print("\n" + "=" * 60)
    print("SPACE-SEPARATED tokenization (current training format):")
    print("=" * 60)
    spaced_dist = Counter()
    for lbl in sample_labels:
        spaced = " ".join(lbl)
        ids = tok(spaced, return_tensors=None).input_ids
        pieces = tok.convert_ids_to_tokens(ids)
        decoded = tok.decode(ids, skip_special_tokens=True).replace(" ", "")
        ok = "OK" if len(ids) == 7 else f"WRONG (expected 7, got {len(ids)})"
        print(f"  '{spaced}' -> {len(ids)} tokens: {pieces}  ->  decode='{decoded}'  [{ok}]")
        spaced_dist[len(ids)] += 1
    print("\nLength distribution (spaced):", dict(spaced_dist))


if __name__ == "__main__":
    main()
