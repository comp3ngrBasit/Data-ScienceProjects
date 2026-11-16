"""Start the API server:  python run.py  [--host 127.0.0.1] [--port 8000] [--reload]"""
import argparse

import uvicorn

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MMA Fight Intelligence API")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--reload", action="store_true", help="Auto-reload on code changes (development)")
    args = parser.parse_args()
    uvicorn.run("app.main:app", host=args.host, port=args.port, reload=args.reload, log_level="info")
