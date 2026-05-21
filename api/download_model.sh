#!/bin/bash

# Script tải model TrOCR ONNX và thiết lập môi trường cho Termux (Android)
# Khắc phục lỗi: Rust Panic (SSL) và Unsupported Platform (onnxruntime-node)

MODEL_DIR="../onnx_model"
ONNX_DIR="$MODEL_DIR/onnx"
BASE_URL="https://huggingface.co/cavoixanh1806/captcha-trocr-onnx/resolve/main"

echo "[INFO] 1. Thiết lập 'Bản giả' (Shim) cho onnxruntime-node..."
# Khắc phục lỗi không cài được onnxruntime-node trên Android
SHIM_DIR="node_modules/onnxruntime-node"
mkdir -p "$SHIM_DIR"
echo '{"name":"onnxruntime-node","main":"index.js"}' > "$SHIM_DIR/package.json"
echo 'const common = require("onnxruntime-common"); module.exports = common;' > "$SHIM_DIR/index.js"
echo 'import * as common from "onnxruntime-common"; export default common;' > "$SHIM_DIR/index.mjs"

echo "[INFO] 2. Tạo thư mục model..."
mkdir -p "$ONNX_DIR"

echo "[INFO] 3. Bắt đầu tải các file cấu hình..."
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

echo "[INFO] 4. Đang tải các file Model ONNX (Dung lượng lớn, vui lòng đợi)..."

echo " -> Tải encoder_model.onnx (~340MB)..."
curl -L "$BASE_URL/onnx/encoder_model.onnx" -o "$ONNX_DIR/encoder_model.onnx"

echo " -> Tải decoder_model_merged.onnx (~1.2GB)..."
curl -L "$BASE_URL/onnx/decoder_model_merged.onnx" -o "$ONNX_DIR/decoder_model_merged.onnx"

echo "[SUCCESS] Môi trường đã sẵn sàng và đã tải xong model!"
echo "Bây giờ bạn có thể chạy: npm start"
