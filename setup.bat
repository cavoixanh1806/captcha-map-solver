@echo off
REM Setup script for Windows (Win10/11) with NVIDIA GPU.
REM Usage:
REM   Double-click setup.bat or run it in CMD/PowerShell.

echo ===================================================
echo === Step 1: Checking System Dependencies ===
echo ===================================================

REM Check Git
git --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [INFO] Git is NOT installed. Attempting to install via winget...
    winget install --id Git.Git -e --source winget
    if %errorlevel% equ 0 (
        echo.
        echo [SUCCESS] Git installed successfully!
        echo [WARNING] Please CLOSE this command prompt window, open a NEW one, and run setup.bat again to continue.
        pause
        exit /b
    ) else (
        echo [ERROR] Failed to install Git via winget. Please download and install Git manually from: https://git-scm.com/
        pause
        exit /b
    )
) else (
    echo [OK] Git is already installed.
)

REM Check Python
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [INFO] Python is NOT installed. Attempting to install Python 3.11 via winget...
    winget install --id Python.Python.3.11 -e --source winget --silent --accept-package-agreements --accept-source-agreements
    if %errorlevel% equ 0 (
        echo.
        echo [SUCCESS] Python 3.11 installed successfully!
        echo [INFO] Refreshing PATH environment variable so you don't need to restart this window...
        
        for /f "tokens=2*" %%A in ('reg query "HKLM\System\CurrentControlSet\Control\Session Manager\Environment" /v Path 2^>nul') do set "SYS_PATH=%%B"
        for /f "tokens=2*" %%A in ('reg query "HKCU\Environment" /v Path 2^>nul') do set "USER_PATH=%%B"
        set "PATH=%USER_PATH%;%SYS_PATH%"
        
        python --version >nul 2>&1
        if %errorlevel% equ 0 (
            echo [OK] Python is now active in this window! Continuing setup...
        ) else (
            echo [WARNING] Python is installed but cannot be accessed in this current window yet.
            echo Please CLOSE this command prompt window, open a NEW one, and run setup.bat again to continue.
            pause
            exit /b
        )
    ) else (
        echo [ERROR] Failed to install Python via winget. Please download and install Python 3.11 manually from: https://www.python.org/
        pause
        exit /b
    )
) else (
    echo [OK] Python is already installed.
)

echo.
echo ===================================================
echo === Step 2: Creating Virtual Environment ===
echo ===================================================
if not exist .venv (
    echo Creating .venv...
    python -m venv .venv
) else (
    echo Virtual environment .venv already exists.
)

echo.
echo === Upgrading pip ===
.venv\Scripts\python -m pip install --upgrade pip

echo.
echo ===================================================
echo === Step 3: Checking NVIDIA GPU / CUDA ===
echo ===================================================
nvidia-smi >nul 2>&1
if %errorlevel% neq 0 (
    echo [WARNING] NVIDIA GPU/CUDA driver not detected via nvidia-smi.
    echo Defaulting to CPU/standard install or check your GPU drivers.
) else (
    echo [OK] NVIDIA GPU/CUDA driver detected successfully!
)

echo.
echo ===================================================
echo === Step 4: Installing PyTorch (CUDA Optimized) ===
echo ===================================================
echo Check your CUDA version above. Installing PyTorch with CUDA 12.4 support 
echo (fully compatible with CUDA 12.1 up to CUDA 12.8+):
.venv\Scripts\python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

echo.
echo ===================================================
echo === Step 5: Installing Project Requirements ===
echo ===================================================
.venv\Scripts\python -m pip install -r requirements.txt

echo.
echo ===================================================
echo === Setup Complete ===
echo ===================================================
echo Virtual environment created and fully configured!
echo.
echo To start training immediately on RTX 3090/3090 Ti (copy and paste):
echo   .venv\Scripts\python train.py --config configs/trocr_base_3090ti.yaml
echo.
echo Train with Synthetic CAPTCHAs (adds 2,000 synthetic images per epoch):
echo   .venv\Scripts\python train.py --config configs/trocr_base_3090ti.yaml --synth 2000
echo.
echo For standard RTX 3060:
echo   .venv\Scripts\python train.py --config configs/trocr_base.yaml
echo.
pause
