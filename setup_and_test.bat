@echo off
title CAPTCHA Solver - Auto Setup and Test Tool
cd /d "%~dp0"

echo =======================================================
echo   CAPTCHA AI Solver - Automatic Setup and Test Tool
echo =======================================================
echo.

:: 1. Verify Python installation
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python is not installed or not added to your Windows PATH!
    echo Please download and install Python 3.10 first.
    pause
    exit /b
)

:: 2. Create virtual environment if it doesn't exist
if not exist ".venv\Scripts\python.exe" (
    echo [1/3] Creating virtual environment (.venv)...
    python -m venv .venv
    if %errorlevel% neq 0 (
        echo [ERROR] Failed to create virtual environment!
        pause
        exit /b
    )
    echo [SUCCESS] Virtual environment created.
) else (
    echo [1/3] Virtual environment (.venv) already exists.
)

:: 3. Activate and install dependencies if needed
echo [2/3] Verifying and installing required AI dependencies...
:: Check if torch is already installed
.\.venv\Scripts\python.exe -c "import torch, transformers, pytorch_lightning" >nul 2>&1
if %errorlevel% neq 0 (
    echo Dependencies missing. Starting automatic installation...
    echo Installing lightweight PyTorch (CPU optimized)...
    .\.venv\Scripts\pip.exe install torch torchvision --index-url https://download.pytorch.org/whl/cpu
    echo Installing other AI requirements...
    .\.venv\Scripts\pip.exe install transformers pytorch-lightning tqdm pillow
    echo [SUCCESS] Dependencies installed successfully!
) else (
    echo [SUCCESS] All dependencies are already verified.
)

:: 4. Run test prediction
echo.
echo [3/3] Running CAPTCHA Solver test on data/map_00000.png...
echo =======================================================
if exist "data\map_00000.png" (
    .\.venv\Scripts\python.exe solve_captcha.py "data\map_00000.png"
) else (
    echo [WARN] Test image data/map_00000.png was not found.
    echo Please pass any image path manually like: solve_captcha.py [image_path]
)
echo =======================================================
echo.
echo Setup and Test completed successfully!
pause
