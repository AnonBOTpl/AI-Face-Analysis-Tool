@echo off
chcp 65001 >nul
title AI Face Analysis Tool v2

cd /d "%~dp0"

if not exist "venv\Scripts\python.exe" (
    echo [BLAD] Nie znaleziono venv. Uruchom najpierw install.bat
    pause
    exit /b 1
)

call venv\Scripts\activate.bat
python -m src.main

if %errorlevel% neq 0 (
    echo [ERROR] Aplikacja zakonczyla sie bledem.
    pause
)
