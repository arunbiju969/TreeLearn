@echo off
setlocal
cd /d "%~dp0"
echo Starting TreeLearn Segmentation Pipeline with CUDA...

set "PYTHON_EXE=%LOCALAPPDATA%\TreeLearn_venv\Scripts\python.exe"
if not exist "%PYTHON_EXE%" (
    if exist "%~dp0.venv\Scripts\python.exe" (
        set "PYTHON_EXE=%~dp0.venv\Scripts\python.exe"
    ) else (
        set "PYTHON_EXE=python"
    )
)

"%PYTHON_EXE%" tools\pipeline\pipeline.py --config configs\pipeline\pipeline.yaml
pause
