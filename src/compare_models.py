#!/usr/bin/env python
"""
compare_models.py
=================
So sánh độ chính xác của 2 checkpoint TrOCR trên một thư mục ảnh đã gán nhãn.
Nhãn được đọc từ file CSV metadata (data/metadata.csv).

Cách chạy:
    python src/compare_models.py --ckpt1 best-epoch032.ckpt --ckpt2 best-epoch072.ckpt --dir ac
"""

import argparse
import glob
import os
import re
import sys
import time

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)

import torch
from PIL import Image


LABEL_PATTERN = re.compile(r"^map_([A-Z0-9]{5})_.*\.png$", re.IGNORECASE)


def load_trocr(ckpt_path: str, device: str):
    """
    Nạp TrOCR checkpoint, tự động remap tên tham số cũ → mới nếu cần
    để tương thích giữa các phiên bản transformers khác nhau.
    """
    from transformers import TrOCRProcessor, VisionEncoderDecoderModel
    import types

    print(f"   Đang nạp: {ckpt_path} ...")
    t0 = time.time()

    # Bước 1: Đọc raw checkpoint
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    hparams = ckpt.get("hyper_parameters", {})
    cfg = hparams.get("cfg", {})
    pretrained = cfg.get("solver", {}).get("pretrained_model", "microsoft/trocr-base-printed")

    # Bước 2: Tạo processor + model HuggingFace từ đầu (kiến trúc phiên bản hiện tại)
    processor = TrOCRProcessor.from_pretrained(pretrained)
    hf_model = VisionEncoderDecoderModel.from_pretrained(pretrained)

    # Bước 3: Thử load state_dict trực tiếp trước
    state_dict = ckpt["state_dict"]
    # State dict của Lightning có prefix "model." → xóa đi để match với hf_model
    hf_sd = {k.replace("model.", "", 1): v for k, v in state_dict.items() if k.startswith("model.")}

    missing, unexpected = hf_model.load_state_dict(hf_sd, strict=False)

    if len(missing) > 0:
        print(f"   ⚠ {len(missing)} keys bị thiếu — thử remapping (hỗ trợ cả 2 chiều cũ <-> mới)...")
        # Cách 1: Thử remap cũ -> mới (checkpoint cũ, model mới)
        remapped_1 = {}
        for k, v in hf_sd.items():
            new_k = k.replace("encoder.encoder.layer.", "encoder.layers.")
            new_k = new_k.replace(".attention.attention.query.", ".attention.q_proj.")
            new_k = new_k.replace(".attention.attention.key.",   ".attention.k_proj.")
            new_k = new_k.replace(".attention.attention.value.", ".attention.v_proj.")
            new_k = new_k.replace(".attention.output.dense.",    ".attention.o_proj.")
            new_k = new_k.replace(".intermediate.dense.",        ".mlp.fc1.")
            new_k = re.sub(r"(layers\.\d+)\.output\.dense\.", r"\1.mlp.fc2.", new_k)
            remapped_1[new_k] = v
        
        missing1, _ = hf_model.load_state_dict(remapped_1, strict=False)
        
        if len(missing1) < len(missing):
            print(f"   ✅ Remap Cũ -> Mới thành công! Còn {len(missing1)} keys thiếu.")
        else:
            # Cách 2: Thử remap mới -> cũ (checkpoint mới, model cũ)
            remapped_2 = {}
            for k, v in hf_sd.items():
                new_k = k.replace("encoder.layers.", "encoder.encoder.layer.")
                new_k = new_k.replace(".attention.q_proj.", ".attention.attention.query.")
                new_k = new_k.replace(".attention.k_proj.", ".attention.attention.key.")
                new_k = new_k.replace(".attention.v_proj.", ".attention.attention.value.")
                new_k = new_k.replace(".attention.o_proj.", ".attention.output.dense.")
                new_k = new_k.replace(".mlp.fc1.",          ".intermediate.dense.")
                new_k = re.sub(r"(layer\.\d+)\.mlp\.fc2\.", r"\1.output.dense.", new_k)
                remapped_2[new_k] = v
            
            missing2, _ = hf_model.load_state_dict(remapped_2, strict=False)
            if len(missing2) < len(missing):
                print(f"   ✅ Remap Mới -> Cũ thành công! Còn {len(missing2)} keys thiếu.")
            else:
                print(f"   ❌ Cả hai chiều remap đều không hiệu quả ({len(missing2)} keys thiếu).")
    else:
        print(f"   ✅ Load state_dict OK, không cần remap")

    hf_model.eval().to(device)

    # Bước 4: Tạo wrapper đơn giản với .processor và .model như predict.py gốc
    wrapper = types.SimpleNamespace()
    wrapper.processor = processor
    wrapper.model = hf_model

    print(f"   → Nạp xong trong {time.time()-t0:.1f}s\n")
    return wrapper


def predict_one(model, img_path: str, device: str) -> str:
    img = Image.open(img_path).convert("RGB")
    pixel_values = model.processor(images=img, return_tensors="pt").pixel_values.to(device)
    with torch.no_grad():
        gen = model.model.generate(pixel_values)
    return model.processor.batch_decode(gen, skip_special_tokens=True)[0].replace(" ", "").upper()


def load_labels(csv_path: str) -> dict:
    """Đọc nhãn từ file CSV metadata, trả về dict {filename: label}."""
    import csv as _csv
    labels = {}
    with open(csv_path, encoding="utf-8") as f:
        for row in _csv.DictReader(f):
            labels[row["filename"]] = row["text"].strip().upper()
    return labels


def evaluate(model, image_paths: list, device: str, label_map: dict, label: str):
    correct = 0
    total = 0
    results = []
    for img_path in image_paths:
        filename = os.path.basename(img_path)
        true_label = label_map.get(filename)
        if true_label is None:
            continue  # ảnh không có nhãn trong CSV → bỏ qua
        t0 = time.time()
        pred = predict_one(model, img_path, device)
        elapsed_ms = (time.time() - t0) * 1000
        ok = pred == true_label
        if ok:
            correct += 1
        total += 1
        results.append((filename, true_label, pred, ok, elapsed_ms))

    return correct, total, results


def print_comparison(results1, results2, ckpt1_name, ckpt2_name):
    w = 38
    print(f"\n{'STT':<5} | {'Tên File':<{w}} | {'Thực tế':<8} | {ckpt1_name[:12]:<14} | {ckpt2_name[:12]:<14} | {'Thắng'}")
    print("-" * (5 + 3 + w + 3 + 8 + 3 + 14 + 3 + 14 + 3 + 6))

    map2 = {r[0]: r for r in results2}
    wins = {"model1": 0, "model2": 0, "both": 0, "none": 0}

    for idx, (fname, true_lbl, pred1, ok1, ms1) in enumerate(results1, 1):
        if fname not in map2:
            continue
        _, _, pred2, ok2, ms2 = map2[fname]

        if ok1 and ok2:
            w_str = "CẢ HAI"
            wins["both"] += 1
        elif ok1 and not ok2:
            w_str = f"← {ckpt1_name[:8]}"
            wins["model1"] += 1
        elif ok2 and not ok1:
            w_str = f"→ {ckpt2_name[:8]}"
            wins["model2"] += 1
        else:
            w_str = "NONE"
            wins["none"] += 1

        print(f"{idx:<5} | {fname[:w]:<{w}} | {true_lbl:<8} | {pred1:<14} | {pred2:<14} | {w_str}")

    return wins


def main():
    parser = argparse.ArgumentParser(description="So sánh 2 checkpoint TrOCR trên bộ dữ liệu test")
    parser.add_argument("--ckpt1", required=True, help="Đường dẫn đến checkpoint thứ nhất")
    parser.add_argument("--ckpt2", required=True, help="Đường dẫn đến checkpoint thứ hai")
    parser.add_argument("--dir", default="ac", help="Thư mục ảnh test")
    parser.add_argument("--csv", default="data/metadata.csv", help="File CSV chứa nhãn (filename,text)")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[*] Thiết bị: {device.upper()}")

    # Đọc nhãn từ CSV
    label_map = load_labels(args.csv)
    print(f"[*] Đã đọc {len(label_map)} nhãn từ '{args.csv}'")

    image_paths = sorted(glob.glob(os.path.join(args.dir, "*.png")))
    labeled_paths = [p for p in image_paths if os.path.basename(p) in label_map]
    print(f"[*] Tìm thấy {len(image_paths)} ảnh trong '{args.dir}', có nhãn: {len(labeled_paths)} ảnh")

    ckpt1_name = os.path.basename(args.ckpt1).replace(".ckpt", "")
    ckpt2_name = os.path.basename(args.ckpt2).replace(".ckpt", "")

    # ---------- Model 1 ----------
    print(f"\n[1/2] Đang đánh giá Model 1: {ckpt1_name}")
    model1 = load_trocr(args.ckpt1, device)
    correct1, total1, results1 = evaluate(model1, labeled_paths, device, label_map, ckpt1_name)
    del model1
    if device == "cuda":
        torch.cuda.empty_cache()

    # ---------- Model 2 ----------
    print(f"\n[2/2] Đang đánh giá Model 2: {ckpt2_name}")
    model2 = load_trocr(args.ckpt2, device)
    correct2, total2, results2 = evaluate(model2, labeled_paths, device, label_map, ckpt2_name)
    del model2
    if device == "cuda":
        torch.cuda.empty_cache()

    # ---------- Bảng so sánh chi tiết ----------
    wins = print_comparison(results1, results2, ckpt1_name, ckpt2_name)

    # ---------- Tóm tắt ----------
    acc1 = correct1 / total1 * 100 if total1 else 0
    acc2 = correct2 / total2 * 100 if total2 else 0

    print("\n" + "=" * 60)
    print(f"  KẾT QUẢ TỔNG HỢP trên {total1} ảnh có nhãn")
    print("=" * 60)
    print(f"  {ckpt1_name:<20} : {correct1:>3}/{total1} đúng  →  {acc1:.1f}%")
    print(f"  {ckpt2_name:<20} : {correct2:>3}/{total2} đúng  →  {acc2:.1f}%")
    print("-" * 60)
    print(f"  Cả hai đúng                : {wins['both']}")
    print(f"  Chỉ {ckpt1_name} đúng   : {wins['model1']}")
    print(f"  Chỉ {ckpt2_name} đúng   : {wins['model2']}")
    print(f"  Cả hai đều sai             : {wins['none']}")
    print("=" * 60)

    if acc1 > acc2:
        print(f"  ✅ Model TỐT HƠN: {ckpt1_name} (+{acc1-acc2:.1f}%)")
    elif acc2 > acc1:
        print(f"  ✅ Model TỐT HƠN: {ckpt2_name} (+{acc2-acc1:.1f}%)")
    else:
        print("  🟡 Hai model có độ chính xác BẰNG NHAU")
    print("=" * 60)


if __name__ == "__main__":
    main()
