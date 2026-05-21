#!/bin/bash

# Script cấu hình môi trường TrOCR cho Termux (Android) - Bản 2.0
# Fix lỗi: [cpu] backend not found

MODEL_DIR="../onnx_model"
ONNX_DIR="$MODEL_DIR/onnx"
BASE_URL="https://huggingface.co/cavoixanh1806/captcha-trocr-onnx/resolve/main"

echo "[INFO] 1. Cài đặt engine Wasm bổ trợ cho Android..."
# Cài đặt thêm onnxruntime-web vì nó chứa các file .wasm cần thiết
npm install onnxruntime-web@1.20.1 --force

echo "[INFO] 2. Thiết lập 'Bản giả' (Shim) thông minh..."
SHIM_DIR="node_modules/onnxruntime-node"
mkdir -p "$SHIM_DIR"
echo '{"name":"onnxruntime-node","main":"index.js"}' > "$SHIM_DIR/package.json"

# Viết mã giả để đánh lừa Transformers.js: Khi nó đòi "cpu", chúng ta đưa cho nó "wasm"
cat <<EOF > "$SHIM_DIR/index.js"
const ort = require('onnxruntime-web');
module.exports = ort;
EOF

echo "[INFO] 3. Tạo thư mục model..."
mkdir -p "$ONNX_DIR"

echo "[INFO] 4. Tải các file cấu hình và Model (sử dụng curl)..."
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
    if [ ! -f "$MODEL_DIR/$file" ]; then
        echo " -> Tải $file..."
        curl -L "$BASE_URL/$file" -o "$MODEL_DIR/$file"
    else
        echo " -> $file đã tồn tại, bỏ qua."
    fi
done

if [ ! -f "$ONNX_DIR/encoder_model.onnx" ]; then
    echo " -> Tải encoder_model.onnx (~340MB)..."
    curl -L "$BASE_URL/onnx/encoder_model.onnx" -o "$ONNX_DIR/encoder_model.onnx"
fi

if [ ! -f "$ONNX_DIR/decoder_model_merged.onnx" ]; then
    echo " -> Tải decoder_model_merged.onnx (~1.2GB)..."
    curl -L "$BASE_URL/onnx/decoder_model_merged.onnx" -o "$ONNX_DIR/decoder_model_merged.onnx"
fi

echo "[SUCCESS] Môi trường Wasm đã sẵn sàng!"
echo "Bây giờ bạn hãy chạy: npm start"
