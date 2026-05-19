@echo off
title CAPTCHA Solver API Launcher
cd /d "%~dp0"

echo ===================================================
echo   Starting CAPTCHA Solver FastAPI Service...
echo ===================================================

:: Check for virtual environment
if not exist ".venv\Scripts\python.exe" (
    echo [ERROR] Virtual environment .venv was not found!
    echo Please make sure you are in the correct TrainAI project folder.
    pause
    exit /b
)

echo [1/2] Activating virtual environment and checking dependencies...
:: Automatically install dependencies if not already done
.\.venv\Scripts\python.exe -c "import fastapi, uvicorn" 2>nul
if %errorlevel% neq 0 (
    echo Dependencies not found. Installing now...
    .\.venv\Scripts\pip.exe install -r api/requirements.txt
) else (
    echo Dependencies already verified.
)

echo [2/2] Launching API...
.\.venv\Scripts\python.exe api/main.py

pause
