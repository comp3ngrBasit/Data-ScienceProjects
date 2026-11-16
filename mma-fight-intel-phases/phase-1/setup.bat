@echo off
setlocal
title MMA Fight Intel - Setup
cd /d "%~dp0"

echo === MMA Fight Intelligence - Phase 1 setup ===
echo.

REM ---- Find Python 3.11+ ----
set "PY="
py -3 -c "import sys; sys.exit(0 if sys.version_info >= (3,11) else 1)" >nul 2>nul
if not errorlevel 1 set "PY=py -3"
if defined PY goto havepy
python -c "import sys; sys.exit(0 if sys.version_info >= (3,11) else 1)" >nul 2>nul
if not errorlevel 1 set "PY=python"
if defined PY goto havepy
echo [ERROR] Python 3.11 or newer was not found.
echo         Install it from https://www.python.org/downloads/  - tick "Add python.exe to PATH"
echo         or run:  winget install Python.Python.3.12
pause
exit /b 1

:havepy
echo Using Python: %PY%

REM ---- Find Node.js ----
where node >nul 2>nul
if not errorlevel 1 goto havenode
echo [ERROR] Node.js was not found. Install Node 22 LTS from https://nodejs.org/
echo         or run:  winget install OpenJS.NodeJS.LTS
pause
exit /b 1

:havenode

echo.
echo [1/4] Creating Python virtual environment in backend\.venv ...
if not exist backend\.venv\Scripts\python.exe %PY% -m venv backend\.venv
if not exist backend\.venv\Scripts\python.exe goto fail

echo [2/4] Installing Python packages ...
call backend\.venv\Scripts\python.exe -m pip install --upgrade pip >nul
call backend\.venv\Scripts\python.exe -m pip install -r backend\requirements.txt || goto fail

echo [3/4] Installing frontend packages ...
pushd frontend
call npm install || (popd & goto fail)

echo [4/4] Building frontend ...
call npm run build || (popd & goto fail)
popd

echo.
echo Setup complete. Run start.bat to launch the app.
pause
exit /b 0

:fail
echo.
echo [ERROR] Setup failed - see the messages above.
pause
exit /b 1
