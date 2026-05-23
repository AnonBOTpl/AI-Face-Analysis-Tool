@echo off
chcp 65001 >nul
title Instalacja AI Face Analysis Tool v2

cd /d "%~dp0"

if exist "venv\Scripts\python.exe" (
    echo [INFO] Venv juz istnieje. Aktualizuje zaleÅ¼noÅci...
) else (
    echo [INFO] Tworzenie wirtualnego srodowiska...
    python -m venv venv
)

call venv\Scripts\activate.bat

echo [INFO] Instalowanie PyTorch z CUDA 12.8...
pip install torch==2.7.1+cu128 torchvision==0.22.1+cu128 --index-url https://download.pytorch.org/whl/cu128

echo [INFO] Instalowanie zaleÅ¼noÅci z requirements.txt...
pip install -r requirements.txt

echo [INFO] Instalowanie MiVOLO v2 (model wieku/pÅci)...
pip install git+https://github.com/WildChlamydia/MiVOLO.git --no-build-isolation

echo [OK] Instalacja zakonczona. Uzyj uruchom.bat aby uruchomic aplikacje.
pause
