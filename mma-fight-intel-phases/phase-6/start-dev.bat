@echo off
REM Development mode: backend with auto-reload + Vite dev server with hot reload.
cd /d "%~dp0"
if not exist backend\.venv\Scripts\python.exe (
  echo Run setup.bat first.
  pause & exit /b 1
)
start "MMA Intel - API (8000)" cmd /k "cd /d %~dp0backend && .venv\Scripts\python.exe run.py --reload"
start "MMA Intel - UI (5173)" cmd /k "cd /d %~dp0frontend && npm run dev"
timeout /t 4 >nul
start "" http://localhost:5173
