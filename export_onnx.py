#!/usr/bin/env python3
"""
Export TrOCR fine-tuned checkpoint → ONNX → INT8 quantization.

Chạy trên PC (Windows/Linux), KHÔNG chạy trên điện thoại.
Output copy sang điện thoại để dùng với mobile_test_ac_onnx.py.

Yêu cầu (chạy trên PC):
  pip install optimum[onnxruntime] onnxruntime

Output:
  ./trocr-finetuned/   — HuggingFace format (trung gian)
  ./trocr-onnx/        — ONNX float32
  ./trocr-onnx-int8/   — ONNX INT8 ← copy cái này sang điện thoại

Copy sang điện thoại:
  # Qua USB (ADB):
  adb push trocr-onnx-int8/ /sdcard/captcha-solver/trocr-onnx-int8/

  # Trên Termux:
  cp -r /sdcard/captcha-solver/trocr-onnx-int8 ~/captcha-solver/

  # Test:
  python mobile_test_ac_onnx.py --model_dir ./trocr-onnx-int8
"""
from __future__ import annotations

import os
import re
import sys
import time
from pathlib import Path

import torch


# ────────────────────────────────────────────────────────────────────────
# Key remapping (giống mobile_test_ac.py)
# ────────────────────────────────────────────────────────────────────────
_REMAP_RULES: list[tuple[str, str]] = [
    (r"encoder\.encoder\.layer\.(\d+)\.",  r"encoder.layers.\1."),
    (r"\.attention\.attention\.query\.",   ".attention.q_proj."),
    (r"\.attention\.attention\.key\.",     ".attention.k_proj."),
    (r"\.attention\.attention\.value\.",   ".attention.v_proj."),
    (r"\.attention\.output\.dense\.",      ".attention.o_proj."),
    (r"\.intermediate\.dense\.",           ".mlp.fc1."),
    (r"(layers\.\d+)\.output\.dense\.",    r"\1.mlp.fc2."),
    (r"\.attention\.attention\.relative_position_bias_table",
     ".attention.relative_position_bias_table"),
    (r"\.attention\.attention\.relative_position_index",
     ".attention.relative_position_index"),
    (r"\.lambda_1$", ".layer_scale1"),
    (r"\.lambda_2$", ".layer_scale2"),
]

def _remap_beit_keys(sd: dict) -> dict:
    needs = any(
        "encoder.encoder.layer." in k or ".attention.attention.query." in k
        for k in sd.keys()
    )
    if not needs:
        return sd
    print("⚙️  Remap BEiT keys (old → new naming)...")
    new_sd, remapped = {}, 0
    for k, v in sd.items():
        new_k = k
        for pattern, replacement in _REMAP_RULES:
            new_k = re.sub(pattern, replacement, new_k)
        if new_k != k:
            remapped += 1
        new_sd[new_k] = v
    print(f"   ✅ Remapped {remapped}/{len(sd)} keys")
    return new_sd


def load_finetuned_model(ckpt_path: str, base_model_id: str = "microsoft/trocr-base-printed"):
    """Load fine-tuned weights từ .ckpt vào VisionEncoderDecoderModel."""
    from transformers import TrOCRProcessor, VisionEncoderDecoderModel

    print(f"\n[1/4] Load checkpoint: {ckpt_path}")
    ckpt_data = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    raw_sd: dict = ckpt_data.get("state_dict", ckpt_data)
    _skip = ("optimizer_states", "lr_schedulers", "hparams",
             "epoch", "global_step", "pytorch-lightning_version")
    sd = {
        (k[len("model."):] if k.startswith("model.") else k): v
        for k, v in raw_sd.items()
        if not any(k.startswith(p) for p in _skip)
    }
    print(f"   {len(sd)} weight tensors")

    sd = _remap_beit_keys(sd)

    print(f"\n[2/4] Load base model: {base_model_id}")
    processor = TrOCRProcessor.from_pretrained(base_model_id)
    model     = VisionEncoderDecoderModel.from_pretrained(base_model_id)

    model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
    model.config.pad_token_id           = processor.tokenizer.pad_token_id
    model.config.eos_token_id           = processor.tokenizer.sep_token_id
    model.config.vocab_size             = model.config.decoder.vocab_size

    result = model.load_state_dict(sd, strict=False)
    _ok = {"encoder.pooler.dense.weight", "encoder.pooler.dense.bias"}
    real_missing = [k for k in result.missing_keys if k not in _ok]
    if real_missing:
        print(f"⚠️  {len(real_missing)} keys missing! Fine-tuned weights not fully loaded.")
    else:
        print("   ✅ All fine-tuned weights loaded")

    model.eval()
    return processor, model


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Export TrOCR checkpoint → ONNX INT8")
    parser.add_argument("--ckpt",        default="best-epoch072.ckpt")
    parser.add_argument("--base_model",  default="microsoft/trocr-base-printed")
    parser.add_argument("--out_hf",      default="./trocr-finetuned",
                        help="Thư mục HuggingFace format (trung gian)")
    parser.add_argument("--out_onnx",    default="./trocr-onnx",
                        help="Thư mục ONNX float32")
    parser.add_argument("--out_int8",    default="./trocr-onnx-int8",
                        help="Thư mục ONNX INT8 ← copy cái này sang điện thoại")
    parser.add_argument("--skip_hf",     action="store_true",
                        help="Bỏ qua bước lưu HF format (nếu đã có)")
    parser.add_argument("--skip_onnx",   action="store_true",
                        help="Bỏ qua bước export ONNX (nếu đã có)")
    args = parser.parse_args()

    # ── Kiểm tra dependencies ─────────────────────────────────────────
    try:
        from optimum.onnxruntime import ORTModelForVision2Seq, ORTQuantizer
        from optimum.onnxruntime.configuration import AutoQuantizationConfig
    except ImportError:
        print("❌ Thiếu thư viện optimum:")
        print("   pip install optimum[onnxruntime] onnxruntime")
        sys.exit(1)

    t_total = time.time()

    # ── Step 1 & 2: Load fine-tuned model ────────────────────────────
    processor, model = load_finetuned_model(args.ckpt, args.base_model)

    # ── Step 3: Lưu sang HuggingFace format ──────────────────────────
    if not args.skip_hf:
        print(f"\n[3/4] Lưu HuggingFace format → {args.out_hf}")
        Path(args.out_hf).mkdir(parents=True, exist_ok=True)
        model.save_pretrained(args.out_hf)
        processor.save_pretrained(args.out_hf)
        print(f"   ✅ Saved to {args.out_hf}")

    # ── Step 4: Export sang ONNX ──────────────────────────────────────
    if not args.skip_onnx:
        print(f"\n[4/4a] Export ONNX float32 → {args.out_onnx}")
        print("   (có thể mất 2-5 phút...)")
        t_onnx = time.time()
        ort_model = ORTModelForVision2Seq.from_pretrained(
            args.out_hf,
            export=True,
            opset=17,
        )
        ort_model.save_pretrained(args.out_onnx)
        processor.save_pretrained(args.out_onnx)
        print(f"   ✅ ONNX exported ({time.time()-t_onnx:.0f}s)")

    # ── Step 5: Quantize INT8 ARM64 ───────────────────────────────────
    print(f"\n[4/4b] Quantize INT8 ARM64 → {args.out_int8}")
    t_q = time.time()

    from optimum.onnxruntime import ORTQuantizer
    from optimum.onnxruntime.configuration import AutoQuantizationConfig

    # ARM64 dynamic INT8 quantization (tương thích NNAPI)
    qconfig = AutoQuantizationConfig.arm64(
        is_static=False,   # dynamic quantization — không cần calibration data
        per_channel=False, # per-tensor (tương thích NNAPI rộng hơn)
    )

    Path(args.out_int8).mkdir(parents=True, exist_ok=True)

    # Quantize encoder
    enc_quantizer = ORTQuantizer.from_pretrained(args.out_onnx, file_name="encoder_model.onnx")
    enc_quantizer.quantize(
        save_dir=args.out_int8,
        quantization_config=qconfig,
    )

    # Quantize decoder (merged với kv-cache)
    for fname in ["decoder_model.onnx", "decoder_model_merged.onnx"]:
        fpath = Path(args.out_onnx) / fname
        if fpath.exists():
            dec_quantizer = ORTQuantizer.from_pretrained(args.out_onnx, file_name=fname)
            dec_quantizer.quantize(
                save_dir=args.out_int8,
                quantization_config=qconfig,
            )
            print(f"   ✅ Quantized {fname}")

    # Copy config files
    import shutil
    for f in Path(args.out_onnx).iterdir():
        if f.suffix in (".json", ".txt") or f.name == "special_tokens_map.json":
            shutil.copy2(f, Path(args.out_int8) / f.name)

    processor.save_pretrained(args.out_int8)

    # Tính kích thước
    onnx_size  = sum(f.stat().st_size for f in Path(args.out_onnx).rglob("*.onnx")) / 1e6
    int8_size  = sum(f.stat().st_size for f in Path(args.out_int8).rglob("*.onnx")) / 1e6

    print(f"   ✅ INT8 quantized ({time.time()-t_q:.0f}s)")
    print(f"   ONNX float32: {onnx_size:.0f} MB → INT8: {int8_size:.0f} MB "
          f"({int8_size/onnx_size*100:.0f}%)")

    # ── Hoàn thành ────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"✅ HOÀN THÀNH! Tổng thời gian: {time.time()-t_total:.0f}s")
    print(f"   File cần copy sang điện thoại: {args.out_int8}/")
    print()
    print("📱 Hướng dẫn copy sang điện thoại:")
    print()
    print("  # Qua USB (ADB):")
    print(f"  adb push {args.out_int8}/ /sdcard/captcha-solver/trocr-onnx-int8/")
    print()
    print("  # Trên Termux:")
    print("  cp -r /sdcard/captcha-solver/trocr-onnx-int8 ~/captcha-solver/")
    print()
    print("  # Cài onnxruntime trên điện thoại:")
    print("  pip install onnxruntime optimum")
    print()
    print("  # Chạy test (ONNX, nhanh ~5-10x):")
    print("  python mobile_test_ac_onnx.py --model_dir ./trocr-onnx-int8")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
