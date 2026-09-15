@echo off
chcp 65001 >nul
setlocal EnableExtensions
cd /d "%~dp0"

set "PYTHONUTF8=1"
set "PYTHONDONTWRITEBYTECODE=1"
set "CUDA_VISIBLE_DEVICES=-1"
set "BASE_PYTHON="

echo ============================================================================
echo  Teachable Machine Local Studio v2.1.0
echo  Windows-only Python 3.13 project .venv installer - CPU runtime
echo  No Linux, CUDA, or secondary runtime setup will be requested.
echo ============================================================================
echo.

rem Prefer the official Python launcher.
where py.exe >nul 2>nul
if errorlevel 1 goto TRY_KNOWN_PATHS
py -3.13 "%CD%\scripts\check_python_313.py" --quiet >nul 2>nul
if errorlevel 1 goto TRY_KNOWN_PATHS
echo [INFO] Using Python launcher: py -3.13
py -3.13 -u "%CD%\scripts\install_local.py"
set "RC=%ERRORLEVEL%"
goto FINISH

:TRY_KNOWN_PATHS
call :CHECK_EXE "%LocalAppData%\Programs\Python\Python313\python.exe"
if defined BASE_PYTHON goto RUN_BASE_PYTHON
call :CHECK_EXE "%ProgramFiles%\Python313\python.exe"
if defined BASE_PYTHON goto RUN_BASE_PYTHON
call :CHECK_EXE "%ProgramFiles%\Python\Python313\python.exe"
if defined BASE_PYTHON goto RUN_BASE_PYTHON
call :CHECK_EXE "C:\Python313\python.exe"
if defined BASE_PYTHON goto RUN_BASE_PYTHON

where python.exe >nul 2>nul
if errorlevel 1 goto TRY_PYTHON313
python.exe "%CD%\scripts\check_python_313.py" --quiet >nul 2>nul
if errorlevel 1 goto TRY_PYTHON313
set "BASE_PYTHON=python.exe"
goto RUN_BASE_PYTHON

:TRY_PYTHON313
where python3.13.exe >nul 2>nul
if errorlevel 1 goto NO_PYTHON
python3.13.exe "%CD%\scripts\check_python_313.py" --quiet >nul 2>nul
if errorlevel 1 goto NO_PYTHON
set "BASE_PYTHON=python3.13.exe"
goto RUN_BASE_PYTHON

:CHECK_EXE
if not exist "%~1" exit /b 1
"%~1" "%CD%\scripts\check_python_313.py" --quiet >nul 2>nul
if errorlevel 1 exit /b 1
set "BASE_PYTHON=%~1"
exit /b 0

:RUN_BASE_PYTHON
echo [INFO] Using Python: %BASE_PYTHON%
"%BASE_PYTHON%" -u "%CD%\scripts\install_local.py"
set "RC=%ERRORLEVEL%"
goto FINISH

:FINISH
echo.
if "%RC%"=="0" goto INSTALL_OK
echo [ERROR] Installation failed with exit code %RC%.
echo [ERROR] Send this log file to the teacher:
echo         "%CD%\logs\LATEST_INSTALL.log"
pause
exit /b %RC%

:INSTALL_OK
echo [OK] Installation completed in Windows CPU mode.
echo [NEXT] Double-click 02_START.bat
pause
exit /b 0

:NO_PYTHON
echo [ERROR] Compatible Python was not found.
echo [REQUIRED] Normal 64-bit CPython 3.13.x
echo [TIP] Install Python 3.13 and enable "Add python.exe to PATH".
echo [NOTE] Experimental free-threaded Python 3.13t is not supported.
pause
exit /b 1
