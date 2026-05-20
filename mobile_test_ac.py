#!/usr/bin/env python3
"""
Test model TrOCR trên các ảnh trong thư mục `ac/`.

Tên file trong ac có dạng:  map_XXXXX_<timestamp>_<hash>__cvxxN.png
                                    ^^^^^
                             label chính xác nằm đây

Cách dùng:
  # Test tất cả ảnh trong ac/
  python mobile_test_ac.py

  # Chỉ định thư mục khác
  python mobile_test_ac.py --ac_dir /sdcard/captcha-solver/ac

  # Chỉ định checkpoint
  python mobile_test_ac.py --ckpt /sdcard/captcha-solver/best-epoch072.ckpt

  # Không in màu (khi terminal không hỗ trợ ANSI)
  python mobile_test_ac.py --no_color
"""
from __future__ import annotations

import argparse
import os
import re
import sys
import time
from pathlib import Path

import torch
from PIL import Image


# ─────────────────────────────────────────────
# Màu ANSI (tắt được bằng --no_color)
# ─────────────────────────────────────────────
class C:
    GREEN  = "\033[92m"
    RED    = "\033[91m"
    YELLOW = "\033[93m"
    CYAN   = "\033[96m"
    BOLD   = "\033[1m"
    RESET  = "\033[0m"

NO_COLOR = False

def c(color: str, text: str) -> str:
    if NO_COLOR:
        return text
    return f"{color}{text}{C.RESET}"


# ─────────────────────────────────────────────
# Trích label từ tên file
# map_XXXXX_timestamp_hash__cvxxN.png  →  XXXXX
# ─────────────────────────────────────────────
_LABEL_RE = re.compile(r"^map_([A-Z0-9]{5})_", re.IGNORECASE)

def label_from_filename(filename: str) -> str | None:
    """Trả về label 5 ký tự hoặc None nếu không parse được."""
    m = _LABEL_RE.match(Path(filename).name)
    if m:
        return m.group(1).upper()
    return None


# ─────────────────────────────────────────────
# Load model (1 lần duy nhất)
# ─────────────────────────────────────────────
def load_model(ckpt_path: str):
    """Load TrOCRLitModel từ checkpoint, map về CPU."""
    # Thêm thư mục gốc project vào sys.path để import src.*
    project_root = Path(__file__).parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    from src.trocr import TrOCRLitModel  # noqa: E402

    print(c(C.CYAN, f"⏳ Loading model từ: {ckpt_path}"))
    print(c(C.YELLOW, "   (lần đầu có thể mất 30–120 giây trên điện thoại…)"))
    t0 = time.time()

    model = TrOCRLitModel.load_from_checkpoint(
        ckpt_path,
        map_location="cpu",   # Snapdragon không có CUDA
    )
    model.eval()

    elapsed = time.time() - t0
    print(c(C.GREEN, f"✅ Model loaded trong {elapsed:.1f}s\n"))
    return model


# ─────────────────────────────────────────────
# Inference 1 ảnh
# ─────────────────────────────────────────────
def predict_one(model, image_path: str) -> tuple[str, float]:
    """Trả về (prediction, thời_gian_giây)."""
    t0 = time.time()

    img = Image.open(image_path).convert("RGB")
    pixel_values = model.processor(
        images=img, return_tensors="pt"
    ).pixel_values  # shape [1, 3, H, W]

    with torch.no_grad():
        gen = model.model.generate(
            pixel_values,
            num_beams=8,
            max_length=16,
            repetition_penalty=1.3,
        )

    pred = model.processor.batch_decode(
        gen, skip_special_tokens=True
    )[0].replace(" ", "").upper()

    return pred, time.time() - t0


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
def main() -> None:
    global NO_COLOR

    parser = argparse.ArgumentParser(
        description="Test TrOCR model trên ảnh trong thư mục ac/"
    )
    parser.add_argument(
        "--ckpt",
        default="best-epoch072.ckpt",
        help="Đường dẫn tới file checkpoint .ckpt",
    )
    parser.add_argument(
        "--ac_dir",
        default="ac",
        help="Thư mục chứa ảnh cần test (mặc định: ./ac)",
    )
    parser.add_argument(
        "--no_color",
        action="store_true",
        help="Tắt màu ANSI (dùng khi terminal không hỗ trợ)",
    )
    args = parser.parse_args()

    if args.no_color:
        NO_COLOR = True

    # ── Tìm ảnh trong ac_dir ──
    ac_dir = Path(args.ac_dir)
    if not ac_dir.exists():
        print(c(C.RED, f"❌ Không tìm thấy thư mục: {ac_dir}"))
        print(f"   Hãy đảm bảo bạn đang chạy script từ thư mục gốc project")
        print(f"   hoặc truyền đúng đường dẫn: --ac_dir /path/to/ac")
        sys.exit(1)

    images = sorted(
        p for p in ac_dir.iterdir()
        if p.suffix.lower() in (".png", ".jpg", ".jpeg", ".webp")
    )

    if not images:
        print(c(C.YELLOW, f"⚠️  Không có ảnh nào trong {ac_dir}"))
        sys.exit(0)

    # ── Load model ──
    ckpt = Path(args.ckpt)
    if not ckpt.exists():
        print(c(C.RED, f"❌ Không tìm thấy checkpoint: {ckpt}"))
        sys.exit(1)

    model = load_model(str(ckpt))

    # ── Header bảng ──
    col_w = [35, 7, 7, 5, 7]  # file, label, pred, ok, ms
    header = (
        f"{'File':<{col_w[0]}} "
        f"{'Label':>{col_w[1]}} "
        f"{'Pred':>{col_w[2]}} "
        f"{'':>{col_w[3]}} "
        f"{'ms':>{col_w[4]}}"
    )
    sep = "─" * (sum(col_w) + len(col_w))
    print(c(C.BOLD, header))
    print(c(C.CYAN, sep))

    # ── Chạy từng ảnh ──
    total = correct = skipped = 0
    total_ms = 0.0
    wrong_list: list[tuple[str, str, str]] = []

    for img_path in images:
        label = label_from_filename(img_path.name)
        pred, t = predict_one(model, str(img_path))
        ms = t * 1000
        total_ms += ms

        fname_display = img_path.name
        # Rút gọn tên file nếu quá dài
        if len(fname_display) > col_w[0]:
            fname_display = "…" + fname_display[-(col_w[0] - 1):]

        if label is None:
            # Không có label trong tên file → chỉ in kết quả
            skipped += 1
            row = (
                f"{fname_display:<{col_w[0]}} "
                f"{'?':>{col_w[1]}} "
                f"{pred:>{col_w[2]}} "
                f"{'~':>{col_w[3]}} "
                f"{ms:>{col_w[4]}.0f}"
            )
            print(c(C.YELLOW, row))
        else:
            total += 1
            ok = pred == label
            if ok:
                correct += 1
                mark = c(C.GREEN, "✓")
                row_color = C.GREEN
            else:
                wrong_list.append((img_path.name, label, pred))
                mark = c(C.RED, "✗")
                row_color = C.RED

            row = (
                f"{fname_display:<{col_w[0]}} "
                f"{label:>{col_w[1]}} "
                f"{pred:>{col_w[2]}} "
                f"{mark:>{col_w[3]}} "
                f"{ms:>{col_w[4]}.0f}"
            )
            print(c(row_color if not ok else "", row))

    # ── Tóm tắt kết quả ──
    print(c(C.CYAN, sep))

    if total > 0:
        acc = correct / total * 100
        avg_ms = total_ms / (total + skipped)
        acc_color = C.GREEN if acc >= 90 else (C.YELLOW if acc >= 70 else C.RED)

        print()
        print(c(C.BOLD, "📊 KẾT QUẢ:"))
        print(f"  Tổng ảnh có label : {total}")
        print(f"  Đúng              : {c(C.GREEN, str(correct))}")
        print(f"  Sai               : {c(C.RED, str(total - correct))}")
        if skipped:
            print(f"  Không có label    : {c(C.YELLOW, str(skipped))}")
        print(f"  Accuracy          : {c(acc_color, f'{acc:.1f}%')}")
        print(f"  Tốc độ TB         : {avg_ms:.0f} ms/ảnh")

        if wrong_list:
            print()
            print(c(C.BOLD + C.RED, "❌ Ảnh đoán sai:"))
            for fname, lbl, p in wrong_list:
                print(f"  {fname}")
                print(f"    Label: {c(C.GREEN, lbl)}  →  Pred: {c(C.RED, p)}")
    else:
        print(c(C.YELLOW, "⚠️  Không có ảnh nào có label để đánh giá."))
        if skipped:
            print(f"   {skipped} ảnh đã được predict nhưng không có ground truth.")


if __name__ == "__main__":
    main()
