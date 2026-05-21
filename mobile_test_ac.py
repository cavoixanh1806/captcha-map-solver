#!/usr/bin/env python3
"""
Test TrOCR trên ảnh trong thư mục ac/ — tối ưu cho Snapdragon 8s Gen 3.

Tốc độ dự kiến:
  --beams 1 --threads 4 --int8  →  ~1-2s/ảnh  (khuyến nghị)
  --beams 8 --threads 4 --int8  →  ~5-6s/ảnh  (chính xác nhất)
  ONNX int8 (xem mobile_test_ac_onnx.py) → <1s/ảnh (nhanh nhất)

Cách dùng:
  python mobile_test_ac.py                        # mặc định: beams=1, threads=4, int8
  python mobile_test_ac.py --beams 8              # full accuracy (chậm hơn)
  python mobile_test_ac.py --no_int8              # tắt quantization
  python mobile_test_ac.py --threads 8            # dùng tất cả 8 core
  python mobile_test_ac.py --offline              # không cần internet
  python mobile_test_ac.py --model_dir ./trocr    # base model local
  python mobile_test_ac.py --ac_dir /sdcard/ac    # thư mục ảnh khác
  python mobile_test_ac.py --no_color             # tắt màu ANSI
"""
from __future__ import annotations

# ═══════════════════════════════════════════════════════════════════════
# FIX ANDROID/TERMUX — set TRƯỚC KHI import bất kỳ thứ gì.
# ═══════════════════════════════════════════════════════════════════════
import os
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")      # tắt xet → HTTPS thường
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import argparse
import re
import sys
import time
from pathlib import Path

import torch

# ARM/QNNPACK backend — tối ưu INT8 cho Snapdragon/ARM
torch.backends.quantized.engine = "qnnpack"

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
    DIM    = "\033[2m"
    RESET  = "\033[0m"

NO_COLOR = False

def col(color: str, text: str) -> str:
    return text if NO_COLOR else f"{color}{text}{C.RESET}"


# ────────────────────────────────────────────────────────────────────────
# Trích label từ tên file:  map_XXXXX_<ts>_<hash>__cvxxN.png → XXXXX
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
# Chú ý: tất cả rules đều anchor vào ^encoder để chỉ áp dụng cho
# encoder BEiT — KHÔNG ảnh hưởng decoder (GPT2/RoBERTa) keys.
# ────────────────────────────────────────────────────────────────────────
_REMAP_RULES: list[tuple[str, str]] = [
    # [1] encoder.encoder.layer.N. → encoder.layers.N.
    (r"^encoder\.encoder\.layer\.(\d+)\.",               r"encoder.layers.\1."),
    # [2-4] attention sub-keys (encoder-only, anchor ^encoder.layers)
    (r"^(encoder\.layers\.\d+)\.attention\.attention\.query\.",  r"\1.attention.q_proj."),
    (r"^(encoder\.layers\.\d+)\.attention\.attention\.key\.",    r"\1.attention.k_proj."),
    (r"^(encoder\.layers\.\d+)\.attention\.attention\.value\.",  r"\1.attention.v_proj."),
    # [5] attention output → o_proj (encoder-only)
    (r"^(encoder\.layers\.\d+)\.attention\.output\.dense\.",     r"\1.attention.o_proj."),
    # [6] FFN fc1 — intermediate.dense (encoder-only, NOT decoder)
    (r"^(encoder\.layers\.\d+)\.intermediate\.dense\.",          r"\1.mlp.fc1."),
    # [7] FFN fc2 — layer output dense (encoder-only)
    (r"^(encoder\.layers\.\d+)\.output\.dense\.",                r"\1.mlp.fc2."),
    # [8-9] relative position bias (encoder-only)
    (r"^(encoder\.layers\.\d+\.attention)\.attention\.relative_position_bias_table",
     r"\1.relative_position_bias_table"),
    (r"^(encoder\.layers\.\d+\.attention)\.attention\.relative_position_index",
     r"\1.relative_position_index"),
    # [10-11] layer scale params (BEiT encoder-only)
    (r"^(encoder\.layers\.\d+)\.lambda_1$", r"\1.layer_scale1"),
    (r"^(encoder\.layers\.\d+)\.lambda_2$", r"\1.layer_scale2"),
]

def _remap_beit_keys(sd: dict) -> dict:
    """
    Áp dụng TẤT CẢ remap rules cho từng key (không break sớm).
    Một key có thể cần nhiều rules liên tiếp:
      encoder.encoder.layer.0.attention.attention.query.weight
      → (rule 1) encoder.layers.0.attention.attention.query.weight
      → (rule 2) encoder.layers.0.attention.q_proj.weight   ✓
    """
    needs_remap = any(
        "encoder.encoder.layer." in k or ".attention.attention.query." in k
        for k in sd.keys()
    )
    if not needs_remap:
        return sd

    print(col(C.YELLOW, "⚙️  BEiT naming cũ (transformers <4.47) → tự động remap..."))
    new_sd: dict = {}
    remapped = 0
    for k, v in sd.items():
        new_k = k
        for pattern, replacement in _REMAP_RULES:
            new_k = re.sub(pattern, replacement, new_k)   # TẤT CẢ rules, không break
        if new_k != k:
            remapped += 1
        new_sd[new_k] = v

    print(col(C.GREEN, f"   ✅ Đã remap {remapped}/{len(sd)} keys"))
    return new_sd


# ────────────────────────────────────────────────────────────────────────
# Load model — BYPASS PyTorch Lightning, load thẳng VisionEncoderDecoder
# ────────────────────────────────────────────────────────────────────────
def load_model(
    ckpt_path: str,
    model_dir: str | None = None,
    offline: bool = False,
    use_int8: bool = True,
):
    """Trả về (processor, model). Xử lý BEiT key mismatch tự động."""
    from transformers import TrOCRProcessor, VisionEncoderDecoderModel

    BASE_MODEL_ID = "microsoft/trocr-base-printed"

    # ── Chọn nguồn base model ──────────────────────────────────────────
    if model_dir:
        pretrained_src = str(Path(model_dir))
        if not Path(pretrained_src).exists():
            print(col(C.RED, f"❌ --model_dir không tồn tại: {model_dir}"))
            sys.exit(1)
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        print(col(C.CYAN, f"📁 Base model local: {pretrained_src}"))
    else:
        cached = _find_cached_model(BASE_MODEL_ID)
        if cached:
            print(col(C.GREEN, "✅ Base model có trong cache"))
            if offline:
                os.environ["TRANSFORMERS_OFFLINE"] = "1"
        else:
            if offline:
                print(col(C.RED, "❌ --offline nhưng chưa có cache. Chạy lần đầu không --offline."))
                sys.exit(1)
            print(col(C.YELLOW, "⬇️  Download base model (~1.3GB, chỉ lần đầu)…"))
        pretrained_src = BASE_MODEL_ID

    # ── B1: Load checkpoint ────────────────────────────────────────────
    print(col(C.CYAN, f"⏳ Đọc checkpoint: {ckpt_path}"))
    t0 = time.time()
    ckpt_data = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    raw_sd: dict = ckpt_data.get("state_dict", ckpt_data)
    _skip = ("optimizer_states", "lr_schedulers", "hparams", "epoch",
             "global_step", "pytorch-lightning_version")
    sd: dict = {}
    for k, v in raw_sd.items():
        if any(k.startswith(p) for p in _skip):
            continue
        sd[k[len("model."):] if k.startswith("model.") else k] = v

    print(col(C.DIM, f"   {len(sd)} weight tensors"))

    # ── B2: Remap BEiT keys ────────────────────────────────────────────
    sd = _remap_beit_keys(sd)

    # ── B3: Load base model ────────────────────────────────────────────
    print(col(C.CYAN, "   Load base model architecture..."))
    processor = TrOCRProcessor.from_pretrained(pretrained_src)
    model     = VisionEncoderDecoderModel.from_pretrained(pretrained_src)

    model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
    model.config.pad_token_id           = processor.tokenizer.pad_token_id
    model.config.eos_token_id           = processor.tokenizer.sep_token_id
    model.config.vocab_size             = model.config.decoder.vocab_size

    # ── B4: Load fine-tuned weights ────────────────────────────────────
    result = model.load_state_dict(sd, strict=False)

    _ok_missing = {"encoder.pooler.dense.weight", "encoder.pooler.dense.bias"}
    real_missing = [k for k in result.missing_keys if k not in _ok_missing]

    if real_missing:
        print(col(C.RED, f"\n{'='*60}"))
        print(col(C.RED, f"❌ CÒN {len(real_missing)} WEIGHTS CHƯA LOAD ĐƯỢC!"))
        print(col(C.RED, f"   Encoder đang dùng base weights → KẾT QUẢ SAI!"))
        print(col(C.YELLOW, f"   → Chạy: git pull origin main  rồi chạy lại"))
        for k in real_missing[:3]:
            print(col(C.DIM, f"   missing: {k}"))
        print(col(C.RED, f"{'='*60}\n"))
    else:
        print(col(C.GREEN, "   ✅ Fine-tuned weights: OK"))

    # ── B5: INT8 Dynamic Quantization (opt-in) ───────────────────────────
    if use_int8:
        print(col(C.YELLOW, "   ⚠️  INT8 quantization — có thể giảm accuracy trên một số thiết bị"))
        print(col(C.YELLOW, "      (qnnpack reduce_range bug). Nếu sai nhiều → bỏ flag --int8"))
        t_q = time.time()
        try:
            # torchao (khuyến nghị bởi PyTorch 2.10+)
            import torchao
            from torchao.quantization import quantize_, int8_dynamic_activation_int8_weight
            quantize_(model, int8_dynamic_activation_int8_weight())
            print(col(C.GREEN, f"   ✅ INT8 (torchao) done ({time.time()-t_q:.1f}s)"))
        except ImportError:
            # Fallback: quantize_dynamic cũ (có thể có bug qnnpack)
            model = torch.quantization.quantize_dynamic(   # type: ignore[attr-defined]
                model, {torch.nn.Linear}, dtype=torch.qint8,
            )
            print(col(C.GREEN, f"   ✅ INT8 (legacy) done ({time.time()-t_q:.1f}s)"))

    model.eval()
    elapsed = time.time() - t0
    print(col(C.GREEN, f"\n✅ Model sẵn sàng ({elapsed:.1f}s)\n"))
    return processor, model


# ────────────────────────────────────────────────────────────────────────
# Inference 1 ảnh
# ────────────────────────────────────────────────────────────────────────
def predict_one(
    processor,
    model,
    image_path: str,
    num_beams: int = 1,
) -> tuple[str, float]:
    t0 = time.time()
    img = Image.open(image_path).convert("RGB")
    pixel_values = processor(images=img, return_tensors="pt").pixel_values

    with torch.inference_mode():          # nhanh hơn no_grad()
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
def main() -> None:
    global NO_COLOR

    parser = argparse.ArgumentParser(
        description="Test TrOCR — tối ưu Snapdragon 8s Gen 3",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--ckpt",      default="best-epoch072.ckpt")
    parser.add_argument("--ac_dir",    default="ac")
    parser.add_argument("--model_dir", default=None,
                        help="Base model local (không download)")
    parser.add_argument("--offline",   action="store_true")
    parser.add_argument("--beams",     type=int, default=2,
                        help="1=greedy/fastest | 2=balanced (default) | 8=most accurate")
    parser.add_argument("--threads",   type=int, default=4,
                        help="Snapdragon 8s Gen3: 4 big cores (X4+A720x3)")
    parser.add_argument("--int8",      action="store_true",
                        help="Bật INT8 quantization (nhanh hơn nhưng có thể giảm accuracy)")
    parser.add_argument("--no_color",  action="store_true")
    args = parser.parse_args()

    if args.no_color:
        NO_COLOR = True

    # ── Cấu hình threads cho Snapdragon ─────────────────────────────
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)       # tránh overhead context switching
    torch.set_flush_denormal(True)         # tránh slow denormal float trên ARM
    n_threads = torch.get_num_threads()
    print(col(C.BOLD, f"🔧 Snapdragon config: {n_threads} threads | "
              f"beams={args.beams} | int8={'ON' if args.int8 else 'OFF (float32)'}"))

    # Ước tính tốc độ
    est_per_img = {
        (1, True): 1.5, (1, False): 8,
        (2, True): 2.5, (2, False): 15,
        (8, True): 8,   (8, False): 60,
    }.get((args.beams, args.int8), args.beams * 8)
    print(col(C.DIM, f"   Ước tính: ~{est_per_img:.0f}s/ảnh (float32, CPU)"))
    print(col(C.DIM,  "   Nhanh hơn: python mobile_test_ac_onnx.py (ONNX)"))
    print()

    # ── Tìm ảnh ──────────────────────────────────────────────────────
    ac_dir = Path(args.ac_dir)
    if not ac_dir.exists():
        print(col(C.RED, f"❌ Không tìm thấy: {ac_dir}"))
        sys.exit(1)

    images = sorted(
        p for p in ac_dir.iterdir()
        if p.suffix.lower() in (".png", ".jpg", ".jpeg", ".webp")
    )
    if not images:
        print(col(C.YELLOW, f"⚠️  Không có ảnh trong {ac_dir}"))
        sys.exit(0)

    print(col(C.CYAN, f"📁 {len(images)} ảnh trong {ac_dir}"))

    # ── Load model ────────────────────────────────────────────────────
    ckpt = Path(args.ckpt)
    if not ckpt.exists():
        print(col(C.RED, f"❌ Không tìm thấy checkpoint: {ckpt}"))
        sys.exit(1)

    processor, model = load_model(
        str(ckpt),
        model_dir=args.model_dir,
        offline=args.offline,
        use_int8=args.int8,
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
        acc      = correct / total * 100
        avg_ms   = total_ms / max(1, total + skipped)
        acc_col  = C.GREEN if acc >= 90 else (C.YELLOW if acc >= 70 else C.RED)
        fps      = 1000 / avg_ms if avg_ms > 0 else 0

        print()
        print(col(C.BOLD, "📊 KẾT QUẢ:"))
        print(f"  Tổng ảnh có label  : {total}")
        print(f"  Đúng               : {col(C.GREEN, str(correct))}")
        print(f"  Sai                : {col(C.RED,   str(total - correct))}")
        if skipped:
            print(f"  Không có label     : {col(C.YELLOW, str(skipped))}")
        print(f"  Accuracy           : {col(acc_col, f'{acc:.1f}%')}")
        print(f"  Tốc độ TB          : {avg_ms:.0f} ms/ảnh  ({fps:.2f} ảnh/s)")
        print()
        print(col(C.DIM, f"  Config: beams={args.beams} | threads={args.threads} | "
              f"int8={'OFF' if args.no_int8 else 'ON'}"))
        print(col(C.DIM,  "  Để nhanh hơn nữa: python mobile_test_ac_onnx.py (NPU/ONNX)"))

        if wrong_list:
            print()
            print(col(C.BOLD + C.RED, "❌ Ảnh đoán sai:"))
            for fname, lbl, p in wrong_list:
                print(f"  {fname}")
                print(f"    Label: {col(C.GREEN, lbl)}  →  Pred: {col(C.RED, p)}")
    else:
        print(col(C.YELLOW, "⚠️  Không có ảnh nào có label."))


if __name__ == "__main__":
    main()
