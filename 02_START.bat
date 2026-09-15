@echo off
chcp 65001 >nul
setlocal EnableExtensions
cd /d "%~dp0"

set "PYTHONUTF8=1"
set "PYTHONDONTWRITEBYTECODE=1"
set "TF_USE_LEGACY_KERAS=1"
set "TF_CPP_MIN_LOG_LEVEL=2"
set "CUDA_VISIBLE_DEVICES=-1"
set "VENV_PYTHON=%CD%\.venv\Scripts\python.exe"

if not exist "%VENV_PYTHON%" goto NO_VENV

"%VENV_PYTHON%" -u "%CD%\scripts\start_local.py"
set "RC=%ERRORLEVEL%"
if "%RC%"=="0" exit /b 0

echo.
echo [ERROR] Local Studio stopped with exit code %RC%.
echo [LOG] "%CD%\logs\LATEST.log"
pause
exit /b %RC%

:NO_VENV
echo [ERROR] Project environment was not found:
echo         "%CD%\.venv"
echo [FIX] Run 01_INSTALL.bat first.
echo [RESET] If the environment is damaged, close Local Studio, delete .venv,
echo         and run 01_INSTALL.bat again. Do not delete workspace.
pause
exit /b 1
