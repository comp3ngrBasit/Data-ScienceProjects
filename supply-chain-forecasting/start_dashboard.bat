@echo off
REM Opens the dashboard at http://localhost:8000/dashboard/
REM Keep this window open while you use the dashboard; close it to stop the server.
cd /d "%~dp0"
where python >nul 2>nul
if errorlevel 1 (
  echo Python was not found. Opening the dashboard file directly instead.
  start "" "%~dp0dashboard\index.html"
  pause
  exit /b
)
start "" http://localhost:8000/dashboard/
python -m http.server 8000
