@echo off
title CAPTCHA Solver - Environment Automatic Setup
cd /d "%~dp0"

echo =======================================================
echo   CAPTCHA AI Solver - Environment Automatic Setup
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

:: 3. Install core PyTorch & Deep Learning dependencies
echo [2/3] Installing lightweight PyTorch (CPU optimized)...
.\.venv\Scripts\pip.exe install torch torchvision --index-url https://download.pytorch.org/whl/cpu

echo.
echo [3/3] Installing transformers, pytorch-lightning and web libraries...
.\.venv\Scripts\pip.exe install transformers pytorch-lightning tqdm pillow fastapi uvicorn python-multipart pydantic

echo.
echo =======================================================
echo   [SUCCESS] Environment setup completed perfectly!
echo   All deep learning and API libraries are ready.
echo =======================================================
echo.
pause
