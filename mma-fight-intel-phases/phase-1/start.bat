@echo off
title MMA Fight Intel - Server
cd /d "%~dp0"
if not exist backend\.venv\Scripts\python.exe (
  echo Run setup.bat first.
  pause & exit /b 1
)
if not exist frontend\dist\index.html (
  echo Frontend build not found - building now...
  pushd frontend & call npm run build & popd
)
echo Starting MMA Fight Intelligence on http://127.0.0.1:8000  (Ctrl+C to stop)
echo API docs: http://127.0.0.1:8000/docs
start "" http://127.0.0.1:8000
cd backend
.venv\Scripts\python.exe run.py --host 127.0.0.1 --port 8000
pause
