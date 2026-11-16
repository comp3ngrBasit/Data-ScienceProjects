#!/bin/sh
# macOS/Linux: start the API and the Control Tower locally (PostgreSQL must be running).
set -e
cd "$(dirname "$0")/.."
python scripts/bootstrap.py --if-needed
uvicorn scip.api.main:app --port 8000 &
sleep 4
SCIP_API_URL=http://localhost:8000 streamlit run dashboard/app.py --server.port 8501
