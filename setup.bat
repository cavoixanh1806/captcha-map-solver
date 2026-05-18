@echo off
REM Setup script for Windows (Win10/11) with NVIDIA GPU.
REM Usage:
REM   git clone https://github.com/cavoixanh1806/captcha-map-solver.git
REM   cd captcha-map-solver
REM   setup.bat
REM   .venv\Scripts\activate
REM   python train.py --config configs/trocr_base_3090ti.yaml --synth 2000

echo === Creating virtual environment ===
python -m venv .venv
call .venv\Scripts\activate
python -m pip install --upgrade pip

echo.
echo === Detecting CUDA version ===
for /f "tokens=*" %%i in ('nvidia-smi ^| findstr "CUDA Version"') do set CUDA_LINE=%%i
echo %CUDA_LINE%

echo.
echo === Installing PyTorch ===
echo Check your CUDA version above and pick the right wheel:
echo   CUDA 12.8+  ->  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
echo   CUDA 12.4   ->  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
echo   CUDA 12.1   ->  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
echo   CUDA 11.8   ->  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
echo.
echo Running with cu128 (RTX 3090 Ti / RTX 3060 with CUDA 12.8+):
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

echo.
echo === Installing project requirements ===
pip install -r requirements.txt

echo.
echo === Setup complete ===
echo Activate env : .venv\Scripts\activate
echo.
echo Train commands:
echo   RTX 3090 Ti (no synth) : python train.py --config configs/trocr_base_3090ti.yaml
echo   RTX 3090 Ti (+ synth)  : python train.py --config configs/trocr_base_3090ti.yaml --synth 2000
echo   RTX 3060    (no synth) : python train.py --config configs/trocr_base.yaml
echo   RTX 3060    (+ synth)  : python train.py --config configs/trocr_base.yaml --synth 1000
echo   Resume from checkpoint : python train.py --config configs/trocr_base.yaml --resume checkpoints\trocr-base\best-epoch004.ckpt
