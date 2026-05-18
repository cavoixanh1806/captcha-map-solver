#!/usr/bin/env bash
# Setup script for Linux rental machines (tested on Ubuntu 22.04 + CUDA 12.x).
# Usage:
#   git clone https://github.com/cavoixanh1806/captcha-map-solver.git
#   cd captcha-map-solver
#   bash setup.sh
#   source .venv/bin/activate
#   python train.py --config configs/trocr_base_3090ti.yaml

set -e

echo "=== Creating virtual environment ==="
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip

echo "=== Detecting CUDA version ==="
CUDA_VER=$(nvidia-smi | grep -oP "CUDA Version: \K[0-9]+\.[0-9]+" | head -1)
echo "Detected CUDA: $CUDA_VER"

# Map CUDA version to PyTorch wheel tag
if [[ "$CUDA_VER" == 12.8* ]] || [[ "$CUDA_VER" == 12.7* ]] || [[ "$CUDA_VER" == 12.6* ]]; then
    WHEEL_TAG="cu128"
elif [[ "$CUDA_VER" == 12.4* ]] || [[ "$CUDA_VER" == 12.5* ]]; then
    WHEEL_TAG="cu124"
elif [[ "$CUDA_VER" == 12.1* ]] || [[ "$CUDA_VER" == 12.2* ]] || [[ "$CUDA_VER" == 12.3* ]]; then
    WHEEL_TAG="cu121"
elif [[ "$CUDA_VER" == 11.8* ]]; then
    WHEEL_TAG="cu118"
else
    echo "WARNING: Unknown CUDA $CUDA_VER, defaulting to cu121"
    WHEEL_TAG="cu121"
fi

echo "=== Installing PyTorch (wheel: $WHEEL_TAG) ==="
pip install torch torchvision --index-url "https://download.pytorch.org/whl/${WHEEL_TAG}"

echo "=== Installing project requirements ==="
pip install -r requirements.txt

echo ""
echo "=== Setup complete ==="
echo "Activate env:  source .venv/bin/activate"
echo "Train (3090Ti): python train.py --config configs/trocr_base_3090ti.yaml"
echo "Train (3060):   python train.py --config configs/trocr_base.yaml"
