#!/usr/bin/env python3
"""
Test model TrOCR trên các ảnh trong thư mục `ac/`.

Tên file trong ac có dạng:  map_XXXXX_<timestamp>_<hash>__cvxxN.png
                                    ^^^^^
                             label chính xác nằm đây (5 ký tự)

Cách dùng:
  python mobile_test_ac.py                         # lần đầu: tự download base model
  python mobile_test_ac.py --offline               # dùng cache, không cần internet
  python mobile_test_ac.py --model_dir ./trocr-base-printed  # dùng thư mục local
  python mobile_test_ac.py --ac_dir /sdcard/captcha-solver/ac
  python mobile_test_ac.py --ckpt best-epoch052.ckpt
  python mobile_test_ac.py --no_color              # tắt màu ANSI
"""
from __future__ import annotations

# ═══════════════════════════════════════════════════════════════════════
# FIX ANDROID/TERMUX — phải set TRƯỚC KHI import bất kỳ thứ gì.
#
# hf-xet bị panic trên Android vì rustls-platform-verifier không khởi
# tạo được trong Termux. Tắt xet → dùng HTTPS thường.
# ═══════════════════════════════════════════════════════════════════════
import os
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import argparse
import re
import sys
import time
from pathlib import Path

import torch
from PIL import Image


# ────────────────────────────────────────────────────────────────────────
# Màu ANSI
# ────────────────────────────────────────────────────────────────────────
class C:
    GREEN  = "\033[92m"
    RED    = "\033[91m"
    YELLOW = "\033[93m"
    CYAN   = "\033[96m"
    BOLD   = "\033[1m"
    RESET  = "\033[0m"

NO_COLOR = False

def col(color: str, text: str) -> str:
    return text if NO_COLOR else f"{color}{text}{C.RESET}"


# ────────────────────────────────────────────────────────────────────────
# Trích label từ tên file
# map_XXXXX_<ts>_<hash>__cvxxN.png → XXXXX
# ────────────────────────────────────────────────────────────────────────
_LABEL_RE = re.compile(r"^map_([A-Z0-9]{5})_", re.IGNORECASE)

def label_from_filename(filename: str) -> str | None:
    m = _LABEL_RE.match(Path(filename).name)
    return m.group(1).upper() if m else None


# ────────────────────────────────────────────────────────────────────────
# Kiểm tra HF cache
# ────────────────────────────────────────────────────────────────────────
def _find_cached_model(model_id: str) -> bool:
    cache_root = Path(
        os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")
    ) / "hub"
    folder = "models--" + model_id.replace("/", "--")
    snap_dir = cache_root / folder / "snapshots"
    if snap_dir.exists():
        for snap in snap_dir.iterdir():
            if any(snap.glob("*.safetensors")) or (snap / "pytorch_model.bin").exists():
                return True
    return False


# ────────────────────────────────────────────────────────────────────────
# Key remapping: BEiT cũ (transformers <4.47) → BEiT mới (≥4.47)
#
# CŨ: encoder.encoder.layer.N.attention.attention.query.*
# MỚI: encoder.layers.N.attention.q_proj.*
# ────────────────────────────────────────────────────────────────────────
_REMAP_RULES: list[tuple[str, str]] = [
    # encoder.encoder.layer.N → encoder.layers.N
    (r"encoder\.encoder\.layer\.(\d+)\.", r"encoder.layers.\1."),
    # attention.attention.query/key/value → attention.q_proj/k_proj/v_proj
    (r"\.attention\.attention\.query\.", ".attention.q_proj."),
    (r"\.attention\.attention\.key\.",   ".attention.k_proj."),
    (r"\.attention\.attention\.value\.", ".attention.v_proj."),
    # attention.output.dense → attention.o_proj
    (r"\.attention\.output\.dense\.", ".attention.o_proj."),
    # intermediate.dense → mlp.fc1
    (r"\.intermediate\.dense\.", ".mlp.fc1."),
    # layerN.output.dense → layerN.mlp.fc2  (chỉ layer output, không phải pooler)
    (r"(layers\.\d+)\.output\.dense\.", r"\1.mlp.fc2."),
    # relative_position_bias (nested → flat)
    (r"\.attention\.attention\.relative_position_bias_table",
     ".attention.relative_position_bias_table"),
    (r"\.attention\.attention\.relative_position_index",
     ".attention.relative_position_index"),
    # lambda_1/2 → layer_scale1/2 (BEiT layer scale params)
    (r"\.lambda_1$", ".layer_scale1"),
    (r"\.lambda_2$", ".layer_scale2"),
]

def _remap_beit_keys(sd: dict) -> dict:
    """
    Phát hiện và remap BEiT attention keys nếu checkpoint dùng naming cũ.
    Trả về state_dict đã remap (hoặc nguyên vẹn nếu không cần).
    """
    needs_remap = any(
        "encoder.encoder.layer." in k or ".attention.attention.query." in k
        for k in sd.keys()
    )

    if not needs_remap:
        return sd  # Đã là format mới

    print(col(C.YELLOW, "⚙️  Phát hiện checkpoint dùng BEiT naming cũ (transformers <4.47)"))
    print(col(C.YELLOW, "   → Tự động remap keys sang format mới..."))

    new_sd: dict = {}
    remapped = 0
    for k, v in sd.items():
        new_k = k
        for pattern, replacement in _REMAP_RULES:
            after = re.sub(pattern, replacement, new_k)
            if after != new_k:
                new_k = after
                remapped += 1
                break  # áp dụng rule đầu tiên khớp
        new_sd[new_k] = v

    print(col(C.GREEN, f"   ✅ Đã remap {remapped}/{len(sd)} keys"))
    return new_sd


# ────────────────────────────────────────────────────────────────────────
# Load model — BYPASS PyTorch Lightning hoàn toàn
#
# Lý do bypass: load_from_checkpoint gọi __init__ → from_pretrained với
# version transformers mới (key mới) nhưng checkpoint lưu key cũ → crash.
# Thay vào đó: load state_dict thủ công + remap + load vào model trực tiếp.
# ────────────────────────────────────────────────────────────────────────
def load_model(
    ckpt_path: str,
    model_dir: str | None = None,
    offline: bool = False,
):
    """
    Trả về (processor, model) đã load fine-tuned weights.
    Không cần pytorch-lightning, không cần src/.
    """
    from transformers import TrOCRProcessor, VisionEncoderDecoderModel

    BASE_MODEL_ID = "microsoft/trocr-base-printed"

    # ── Chọn nguồn base model ──────────────────────────────────────────
    if model_dir:
        pretrained_src = str(Path(model_dir))
        if not Path(pretrained_src).exists():
            print(col(C.RED, f"❌ --model_dir không tồn tại: {model_dir}"))
            sys.exit(1)
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        print(col(C.CYAN, f"📁 Dùng base model từ local: {pretrained_src}"))
    else:
        cached = _find_cached_model(BASE_MODEL_ID)
        if cached:
            print(col(C.GREEN, "✅ Base model có trong cache — không cần download"))
            if offline:
                os.environ["TRANSFORMERS_OFFLINE"] = "1"
        else:
            if offline:
                print(col(C.RED, "❌ --offline nhưng chưa có cache. Chạy lần đầu không --offline."))
                sys.exit(1)
            print(col(C.YELLOW, "⬇️  Đang download base model (~1.3 GB, chỉ lần đầu)…"))
            print(col(C.CYAN,   "   XET đã tắt → dùng HTTPS thường (OK trên Termux)"))
        pretrained_src = BASE_MODEL_ID

    # ── B1: Load checkpoint state_dict ────────────────────────────────
    print(col(C.CYAN, f"\n⏳ Đọc checkpoint: {ckpt_path}"))
    t0 = time.time()

    ckpt_data = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    # Lightning lưu dạng dict với key "state_dict"
    raw_sd: dict = ckpt_data.get("state_dict", ckpt_data)

    # Strip prefix "model." mà Lightning thêm vào (TrOCRLitModel.model = VisionEncoderDecoder)
    # Bỏ các key không phải weights (optimizer, epoch, ...)
    non_weight_prefixes = ("optimizer_states", "lr_schedulers", "hparams",
                           "epoch", "global_step", "pytorch-lightning_version")
    sd: dict = {}
    for k, v in raw_sd.items():
        if any(k.startswith(p) for p in non_weight_prefixes):
            continue
        if k.startswith("model."):
            sd[k[len("model."):]] = v   # strip "model." prefix
        else:
            sd[k] = v

    print(col(C.CYAN, f"   {len(sd)} weight tensors trong checkpoint"))

    # ── B2: Remap BEiT keys nếu cần ───────────────────────────────────
    sd = _remap_beit_keys(sd)

    # ── B3: Load processor & model architecture ────────────────────────
    print(col(C.CYAN, "   Load processor + kiến trúc base model..."))
    processor = TrOCRProcessor.from_pretrained(pretrained_src)
    model     = VisionEncoderDecoderModel.from_pretrained(pretrained_src)

    # Set decoder token IDs (giống src/trocr.py __init__)
    model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
    model.config.pad_token_id           = processor.tokenizer.pad_token_id
    model.config.eos_token_id           = processor.tokenizer.sep_token_id
    model.config.vocab_size             = model.config.decoder.vocab_size

    # ── B4: Load fine-tuned weights ────────────────────────────────────
    print(col(C.CYAN, "   Load fine-tuned weights..."))
    result = model.load_state_dict(sd, strict=False)

    # Báo cáo kết quả load
    # pooler.dense MISSING là bình thường (BEiT pooler không dùng trong TrOCR)
    _ok_missing = {"encoder.pooler.dense.weight", "encoder.pooler.dense.bias"}
    real_missing = [k for k in result.missing_keys if k not in _ok_missing]
    unexpected   = result.unexpected_keys

    if real_missing:
        print(col(C.RED, f"   ⚠️  {len(real_missing)} keys bị thiếu (fine-tuned weights không load):"))
        for k in real_missing[:5]:
            print(col(C.RED, f"      - {k}"))
        if len(real_missing) > 5:
            print(col(C.RED, f"      ... và {len(real_missing)-5} keys nữa"))
        print(col(C.YELLOW, "   → Kết quả có thể kém chính xác hơn."))
    else:
        print(col(C.GREEN, "   ✅ Tất cả fine-tuned weights load thành công!"))

    if unexpected:
        # Unexpected keys thường không ảnh hưởng inference
        print(col(C.YELLOW, f"   ℹ️  {len(unexpected)} keys thừa trong checkpoint (bỏ qua)"))

    model.eval()
    elapsed = time.time() - t0
    print(col(C.GREEN, f"\n✅ Model sẵn sàng trong {elapsed:.1f}s\n"))
    return processor, model


# ────────────────────────────────────────────────────────────────────────
# Inference 1 ảnh
# ────────────────────────────────────────────────────────────────────────
def predict_one(processor, model, image_path: str) -> tuple[str, float]:
    t0 = time.time()
    img = Image.open(image_path).convert("RGB")
    pixel_values = processor(images=img, return_tensors="pt").pixel_values

    with torch.no_grad():
        gen = model.generate(
            pixel_values,
            num_beams=8,
            max_length=16,
            repetition_penalty=1.3,
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
def main() -> None:
    global NO_COLOR

    parser = argparse.ArgumentParser(
        description="Test TrOCR model trên ảnh trong thư mục ac/"
    )
    parser.add_argument("--ckpt",      default="best-epoch072.ckpt",
                        help="Checkpoint .ckpt (mặc định: best-epoch072.ckpt)")
    parser.add_argument("--ac_dir",    default="ac",
                        help="Thư mục ảnh test (mặc định: ./ac)")
    parser.add_argument("--model_dir", default=None,
                        help="Base model local (thay vì download HuggingFace)")
    parser.add_argument("--offline",   action="store_true",
                        help="Chạy offline, dùng HF cache")
    parser.add_argument("--no_color",  action="store_true",
                        help="Tắt màu ANSI")
    args = parser.parse_args()

    if args.no_color:
        NO_COLOR = True

    # ── Tìm ảnh ──────────────────────────────────────────────────────
    ac_dir = Path(args.ac_dir)
    if not ac_dir.exists():
        print(col(C.RED, f"❌ Không tìm thấy thư mục: {ac_dir}"))
        print(f"   Truyền đúng đường dẫn: --ac_dir /path/to/ac")
        sys.exit(1)

    images = sorted(
        p for p in ac_dir.iterdir()
        if p.suffix.lower() in (".png", ".jpg", ".jpeg", ".webp")
    )
    if not images:
        print(col(C.YELLOW, f"⚠️  Không có ảnh nào trong {ac_dir}"))
        sys.exit(0)

    # ── Kiểm tra checkpoint ───────────────────────────────────────────
    ckpt = Path(args.ckpt)
    if not ckpt.exists():
        print(col(C.RED, f"❌ Không tìm thấy checkpoint: {ckpt}"))
        sys.exit(1)

    # ── Load model ────────────────────────────────────────────────────
    processor, model = load_model(
        str(ckpt),
        model_dir=args.model_dir,
        offline=args.offline,
    )

    # ── Bảng kết quả ─────────────────────────────────────────────────
    col_w = [35, 7, 7, 5, 7]
    header = (
        f"{'File':<{col_w[0]}} "
        f"{'Label':>{col_w[1]}} "
        f"{'Pred':>{col_w[2]}} "
        f"{'':>{col_w[3]}} "
        f"{'ms':>{col_w[4]}}"
    )
    sep = "─" * (sum(col_w) + len(col_w))
    print(col(C.BOLD, header))
    print(col(C.CYAN, sep))

    total = correct = skipped = 0
    total_ms = 0.0
    wrong_list: list[tuple[str, str, str]] = []

    for img_path in images:
        label = label_from_filename(img_path.name)
        pred, t = predict_one(processor, model, str(img_path))
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
                mark      = col(C.GREEN, "✓")
                row_color = ""
            else:
                wrong_list.append((img_path.name, label, pred))
                mark      = col(C.RED, "✗")
                row_color = C.RED

            row = (
                f"{fname:<{col_w[0]}} "
                f"{label:>{col_w[1]}} "
                f"{pred:>{col_w[2]}} "
                f"{mark:>{col_w[3]}} "
                f"{ms:>{col_w[4]}.0f}"
            )
            print(col(row_color, row))

    # ── Tóm tắt ──────────────────────────────────────────────────────
    print(col(C.CYAN, sep))

    if total > 0:
        acc     = correct / total * 100
        avg_ms  = total_ms / max(1, total + skipped)
        acc_col = C.GREEN if acc >= 90 else (C.YELLOW if acc >= 70 else C.RED)

        print()
        print(col(C.BOLD, "📊 KẾT QUẢ:"))
        print(f"  Tổng ảnh có label : {total}")
        print(f"  Đúng              : {col(C.GREEN, str(correct))}")
        print(f"  Sai               : {col(C.RED,   str(total - correct))}")
        if skipped:
            print(f"  Không có label    : {col(C.YELLOW, str(skipped))}")
        print(f"  Accuracy          : {col(acc_col, f'{acc:.1f}%')}")
        print(f"  Tốc độ TB         : {avg_ms:.0f} ms/ảnh")

        if wrong_list:
            print()
            print(col(C.BOLD + C.RED, "❌ Ảnh đoán sai:"))
            for fname, lbl, p in wrong_list:
                print(f"  {fname}")
                print(f"    Label: {col(C.GREEN, lbl)}  →  Pred: {col(C.RED, p)}")
    else:
        print(col(C.YELLOW, "⚠️  Không có ảnh nào có label để đánh giá."))
        if skipped:
            print(f"   {skipped} ảnh đã predict nhưng không có ground truth.")


if __name__ == "__main__":
    main()
