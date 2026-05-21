#!/bin/bash

# Script tải model TrOCR ONNX cho Termux (Android)
# Khắc phục lỗi Rust Panic (rustls-platform-verifier)

MODEL_DIR="../onnx_model"
ONNX_DIR="$MODEL_DIR/onnx"
BASE_URL="https://huggingface.co/cavoixanh1806/captcha-trocr-onnx/resolve/main"

echo "[INFO] Đang tạo thư mục model..."
mkdir -p "$ONNX_DIR"

echo "[INFO] Bắt đầu tải các file cấu hình..."
files=(
    "config.json"
    "generation_config.json"
    "preprocessor_config.json"
    "special_tokens_map.json"
    "tokenizer.json"
    "tokenizer_config.json"
    "vocab.json"
    "merges.txt"
)

for file in "${files[@]}"; do
    echo " -> Tải $file..."
    curl -L "$BASE_URL/$file" -o "$MODEL_DIR/$file"
done

echo "[INFO] Đang tải các file Model ONNX (Dung lượng lớn, vui lòng đợi)..."

echo " -> Tải encoder_model.onnx (~340MB)..."
curl -L "$BASE_URL/onnx/encoder_model.onnx" -o "$ONNX_DIR/encoder_model.onnx"

echo " -> Tải decoder_model_merged.onnx (~1.2GB)..."
curl -L "$BASE_URL/onnx/decoder_model_merged.onnx" -o "$ONNX_DIR/decoder_model_merged.onnx"

echo "[SUCCESS] Đã tải xong toàn bộ model vào thư mục onnx_model!"
