@echo off
title CAPTCHA Solver API Launcher
cd /d "%~dp0"

echo ===================================================
echo   Starting CAPTCHA Solver FastAPI Service...
echo ===================================================
echo.
echo Select running mode:
echo   [1] CPU Mode (For systems without NVIDIA GPU, or running on CPU)
echo   [2] RTX GPU Mode (For systems with NVIDIA GPU, utilizes CUDA)
echo.
set /p choice="Enter choice (1 or 2, default is 1): "

if "%choice%"=="" set choice=1

:: Check for virtual environment
if not exist ".venv\Scripts\python.exe" (
    echo [INFO] Virtual environment .venv was not found!
    echo Creating virtual environment .venv...
    python -m venv .venv
    if errorlevel 1 (
        echo [ERROR] Failed to create virtual environment!
        pause
        exit /b
    )
    echo [SUCCESS] Virtual environment created.
)

if "%choice%"=="2" goto rtx_mode
goto cpu_mode

:rtx_mode
echo [INFO] Checking environment for RTX GPU Mode...
set CUDA_VISIBLE_DEVICES=0
:: Verify if CUDA-enabled PyTorch is ready
.\.venv\Scripts\python.exe -c "import torch; assert torch.cuda.is_available()" 2>nul
if errorlevel 1 goto install_rtx
echo [SUCCESS] RTX GPU mode verified (CUDA is active).
goto launch_api

:install_rtx
echo [WARNING] CUDA-enabled PyTorch is not found or not working.
echo Installing/Upgrading PyTorch to CUDA 12.4 version...
.\.venv\Scripts\pip.exe install --upgrade --force-reinstall torch torchvision --index-url https://download.pytorch.org/whl/cu124
.\.venv\Scripts\pip.exe install transformers pytorch-lightning tqdm pillow fastapi uvicorn python-multipart pydantic sentencepiece
echo [SUCCESS] RTX GPU dependencies installed successfully!
goto launch_api

:cpu_mode
echo [INFO] Checking environment for CPU Mode...
set CUDA_VISIBLE_DEVICES=-1
:: Verify if PyTorch is installed
.\.venv\Scripts\python.exe -c "import torch" 2>nul
if errorlevel 1 goto install_cpu
echo [SUCCESS] CPU mode verified.
goto launch_api

:install_cpu
echo [WARNING] PyTorch is not found.
echo Installing CPU-optimized PyTorch...
.\.venv\Scripts\pip.exe install torch torchvision --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\pip.exe install transformers pytorch-lightning tqdm pillow fastapi uvicorn python-multipart pydantic sentencepiece
echo [SUCCESS] CPU dependencies installed successfully!
goto launch_api

:launch_api
echo.
echo ===================================================
echo   Launching CAPTCHA Solver API...
echo ===================================================
.\.venv\Scripts\python.exe api/main.py

pause
