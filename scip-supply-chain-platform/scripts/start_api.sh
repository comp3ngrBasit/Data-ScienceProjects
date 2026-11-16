#!/bin/sh
# Bootstrap the platform on first start (ingest -> train -> optimise -> automate), then serve the API.
set -e
python scripts/bootstrap.py --if-needed
exec uvicorn scip.api.main:app --host 0.0.0.0 --port 8000
