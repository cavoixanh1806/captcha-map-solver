@echo off
title CAPTCHA Solver API Launcher
cd /d "%~dp0"

echo ===================================================
echo   Starting CAPTCHA Solver FastAPI Service...
echo ===================================================

:: Check for virtual environment
if not exist ".venv\Scripts\python.exe" (
    echo [INFO] Virtual environment .venv was not found!
    echo Creating virtual environment .venv...
    python -m venv .venv
    if %errorlevel% neq 0 (
        echo [ERROR] Failed to create virtual environment!
        pause
        exit /b
    )
    echo [SUCCESS] Virtual environment created.
)

echo [1/2] Activating virtual environment and checking all dependencies...
:: Automatically verify and install all dependencies if not already done
.\.venv\Scripts\python.exe -c "import torch, transformers, pytorch_lightning, fastapi, uvicorn" 2>nul
if %errorlevel% neq 0 (
    echo Core dependencies not found. Installing everything now...
    
    echo 1. Installing lightweight PyTorch CPU optimized...
    .\.venv\Scripts\pip.exe install torch torchvision --index-url https://download.pytorch.org/whl/cpu
    
    echo 2. Installing Transformers, Lightning and API web libraries...
    .\.venv\Scripts\pip.exe install transformers pytorch-lightning tqdm pillow fastapi uvicorn python-multipart pydantic
    
    echo [SUCCESS] All dependencies installed successfully!
) else (
    echo [SUCCESS] All dependencies already verified.
)

echo [2/2] Launching API...
.\.venv\Scripts\python.exe api/main.py

pause
