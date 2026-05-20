#!/usr/bin/env python3
"""
Test TrOCR bằng ONNX Runtime — tận dụng NPU/GPU Snapdragon 8s Gen 3.

Nhanh hơn mobile_test_ac.py (PyTorch) khoảng 5-10x.

Yêu cầu:
  pip install onnxruntime optimum

Cần chuẩn bị trước:
  - Chạy export_onnx.py trên PC để tạo ./trocr-onnx-int8/
  - Copy thư mục trocr-onnx-int8/ sang điện thoại

Cách dùng:
  python mobile_test_ac_onnx.py --model_dir ./trocr-onnx-int8
  python mobile_test_ac_onnx.py --model_dir ./trocr-onnx-int8 --ac_dir /sdcard/captcha-solver/ac
  python mobile_test_ac_onnx.py --model_dir ./trocr-onnx-int8 --nnapi  # thử NPU (experimental)
"""
from __future__ import annotations

import os
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import argparse
import re
import sys
import time
from pathlib import Path


# ────────────────────────────────────────────────────────────────────────
# Màu ANSI
# ────────────────────────────────────────────────────────────────────────
class C:
    GREEN  = "\033[92m"
    RED    = "\033[91m"
    YELLOW = "\033[93m"
    CYAN   = "\033[96m"
    BOLD   = "\033[1m"
    DIM    = "\033[2m"
    RESET  = "\033[0m"

NO_COLOR = False

def col(color: str, text: str) -> str:
    return text if NO_COLOR else f"{color}{text}{C.RESET}"


_LABEL_RE = re.compile(r"^map_([A-Z0-9]{5})_", re.IGNORECASE)

def label_from_filename(filename: str) -> str | None:
    m = _LABEL_RE.match(Path(filename).name)
    return m.group(1).upper() if m else None


# ────────────────────────────────────────────────────────────────────────
# Load model ONNX
# ────────────────────────────────────────────────────────────────────────
def load_model_onnx(model_dir: str, use_nnapi: bool = False):
    """Load ONNX model với optional NNAPI (NPU Snapdragon)."""
    try:
        from optimum.onnxruntime import ORTModelForVision2Seq
        from transformers import TrOCRProcessor
        import onnxruntime as ort
    except ImportError as e:
        print(col(C.RED, f"❌ Thiếu thư viện: {e}"))
        print(col(C.YELLOW, "   pip install onnxruntime optimum"))
        sys.exit(1)

    # Kiểm tra NNAPI
    available_providers = ort.get_available_providers()
    print(col(C.DIM, f"   Available providers: {available_providers}"))

    if use_nnapi:
        if "NnapiExecutionProvider" in available_providers:
            providers = ["NnapiExecutionProvider", "CPUExecutionProvider"]
            print(col(C.GREEN, "✅ NNAPI (NPU Snapdragon) available!"))
        else:
            providers = ["CPUExecutionProvider"]
            print(col(C.YELLOW, "⚠️  NNAPI không available (cần onnxruntime-android), dùng CPU"))
    else:
        providers = ["CPUExecutionProvider"]

    print(col(C.CYAN, f"⏳ Load ONNX model từ: {model_dir}"))
    t0 = time.time()

    # ORTModelForVision2Seq tự xử lý encoder + decoder loop
    model = ORTModelForVision2Seq.from_pretrained(
        model_dir,
        providers=providers,
        use_merged=True,   # dùng decoder_model_merged.onnx (kv-cache, nhanh hơn)
    )
    processor = TrOCRProcessor.from_pretrained(model_dir)

    # Set generation config
    model.generation_config.decoder_start_token_id = processor.tokenizer.cls_token_id
    model.generation_config.pad_token_id           = processor.tokenizer.pad_token_id
    model.generation_config.eos_token_id           = processor.tokenizer.sep_token_id

    elapsed = time.time() - t0
    backend = "NNAPI+CPU" if use_nnapi and "NnapiExecutionProvider" in available_providers else "CPU (ONNX)"
    print(col(C.GREEN, f"✅ ONNX model loaded ({elapsed:.1f}s) | Backend: {backend}\n"))
    return processor, model


# ────────────────────────────────────────────────────────────────────────
# Inference
# ────────────────────────────────────────────────────────────────────────
def predict_one(processor, model, image_path: str, num_beams: int = 1) -> tuple[str, float]:
    from PIL import Image
    t0 = time.time()
    img = Image.open(image_path).convert("RGB")
    pixel_values = processor(images=img, return_tensors="pt").pixel_values

    gen = model.generate(
        pixel_values,
        num_beams=num_beams,
        max_length=16,
        repetition_penalty=1.3 if num_beams > 1 else 1.0,
        decoder_start_token_id=processor.tokenizer.cls_token_id,
        pad_token_id=processor.tokenizer.pad_token_id,
        eos_token_id=processor.tokenizer.sep_token_id,
        early_stopping=True,
    )
    pred = processor.batch_decode(gen, skip_special_tokens=True)[0].replace(" ", "").upper()
    return pred, time.time() - t0


# ────────────────────────────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────────────────────────────
def main():
    global NO_COLOR

    parser = argparse.ArgumentParser(
        description="Test TrOCR ONNX — tận dụng NPU Snapdragon",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model_dir", required=True,
                        help="Thư mục trocr-onnx-int8/ (export từ export_onnx.py)")
    parser.add_argument("--ac_dir",    default="ac")
    parser.add_argument("--beams",     type=int, default=1,
                        help="1=greedy/nhanh | 2=cân bằng | 8=chính xác")
    parser.add_argument("--nnapi",     action="store_true",
                        help="Thử dùng NNAPI (NPU Snapdragon) — experimental")
    parser.add_argument("--no_color",  action="store_true")
    args = parser.parse_args()

    if args.no_color:
        NO_COLOR = True

    # Kiểm tra model_dir
    model_dir = Path(args.model_dir)
    if not model_dir.exists():
        print(col(C.RED, f"❌ Không tìm thấy model: {model_dir}"))
        print(col(C.YELLOW, "   Chạy export_onnx.py trên PC trước, rồi copy sang điện thoại"))
        sys.exit(1)

    has_onnx = any(model_dir.glob("*.onnx")) or any((model_dir / "encoder_model.onnx").exists()
                                                     for _ in [None])
    if not has_onnx:
        print(col(C.RED, f"❌ Không tìm thấy file .onnx trong {model_dir}"))
        sys.exit(1)

    print(col(C.BOLD, f"🔧 ONNX Runtime | beams={args.beams} | nnapi={args.nnapi}"))

    # Load model
    processor, model = load_model_onnx(str(model_dir), use_nnapi=args.nnapi)

    # Tìm ảnh
    ac_dir = Path(args.ac_dir)
    if not ac_dir.exists():
        print(col(C.RED, f"❌ Không tìm thấy thư mục: {ac_dir}"))
        sys.exit(1)

    images = sorted(
        p for p in ac_dir.iterdir()
        if p.suffix.lower() in (".png", ".jpg", ".jpeg", ".webp")
    )
    if not images:
        print(col(C.YELLOW, f"⚠️  Không có ảnh trong {ac_dir}"))
        sys.exit(0)

    print(col(C.CYAN, f"📁 {len(images)} ảnh trong {ac_dir}"))

    # Bảng kết quả
    col_w = [35, 7, 7, 5, 7]
    header = (
        f"{'File':<{col_w[0]}} "
        f"{'Label':>{col_w[1]}} {'Pred':>{col_w[2]}} {'':>{col_w[3]}} {'ms':>{col_w[4]}}"
    )
    sep = "─" * (sum(col_w) + len(col_w))
    print(col(C.BOLD, header))
    print(col(C.CYAN, sep))

    total = correct = skipped = 0
    total_ms = 0.0
    wrong_list: list[tuple[str, str, str]] = []

    for img_path in images:
        label = label_from_filename(img_path.name)
        pred, t = predict_one(processor, model, str(img_path), num_beams=args.beams)
        ms = t * 1000
        total_ms += ms

        fname = img_path.name
        if len(fname) > col_w[0]:
            fname = "…" + fname[-(col_w[0] - 1):]

        if label is None:
            skipped += 1
            print(col(C.YELLOW,
                f"{fname:<{col_w[0]}} {'?':>{col_w[1]}} {pred:>{col_w[2]}} {'~':>{col_w[3]}} {ms:>{col_w[4]}.0f}"
            ))
        else:
            total += 1
            ok = (pred == label)
            if ok:
                correct += 1
                mark = col(C.GREEN, "✓")
                row_color = ""
            else:
                wrong_list.append((img_path.name, label, pred))
                mark = col(C.RED, "✗")
                row_color = C.RED
            row = (
                f"{fname:<{col_w[0]}} "
                f"{label:>{col_w[1]}} {pred:>{col_w[2]}} {mark:>{col_w[3]}} {ms:>{col_w[4]}.0f}"
            )
            print(col(row_color, row))

    print(col(C.CYAN, sep))

    if total > 0:
        acc     = correct / total * 100
        avg_ms  = total_ms / max(1, total + skipped)
        acc_col = C.GREEN if acc >= 90 else (C.YELLOW if acc >= 70 else C.RED)

        print()
        print(col(C.BOLD, "📊 KẾT QUẢ (ONNX):"))
        print(f"  Tổng ảnh có label : {total}")
        print(f"  Đúng              : {col(C.GREEN, str(correct))}")
        print(f"  Sai               : {col(C.RED,   str(total - correct))}")
        if skipped:
            print(f"  Không có label    : {col(C.YELLOW, str(skipped))}")
        print(f"  Accuracy          : {col(acc_col, f'{acc:.1f}%')}")
        print(f"  Tốc độ TB         : {avg_ms:.0f} ms/ảnh  ({1000/avg_ms:.2f} ảnh/s)")

        if wrong_list:
            print()
            print(col(C.BOLD + C.RED, "❌ Ảnh đoán sai:"))
            for fname, lbl, p in wrong_list:
                print(f"  {fname}")
                print(f"    Label: {col(C.GREEN, lbl)}  →  Pred: {col(C.RED, p)}")


if __name__ == "__main__":
    main()
