# Windows: start the API and the Control Tower locally (PostgreSQL must be running; see README).
# Usage:  powershell -ExecutionPolicy Bypass -File scripts\run_local.ps1
$ErrorActionPreference = "Stop"
Set-Location (Split-Path $PSScriptRoot -Parent)
python scripts/bootstrap.py --if-needed
Start-Process -NoNewWindow python -ArgumentList "-m","uvicorn","scip.api.main:app","--port","8000"
Start-Sleep -Seconds 4
$env:SCIP_API_URL = "http://localhost:8000"
python -m streamlit run dashboard/app.py --server.port 8501
