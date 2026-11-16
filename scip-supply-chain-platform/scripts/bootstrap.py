"""Build the whole platform: ingest datasets -> train models -> optimise -> run automation -> index Copilot.

Usage:
    python scripts/bootstrap.py              # full rebuild (resets the database)
    python scripts/bootstrap.py --if-needed  # only if the database has not been bootstrapped yet (Docker start)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scip.core.db import init_db  # noqa: E402
from scip.services.bootstrap import bootstrap, is_bootstrapped  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--if-needed", action="store_true")
    args = ap.parse_args()
    init_db()
    if args.if_needed and is_bootstrapped():
        print("Platform already bootstrapped — skipping.")
        return
    out = bootstrap(progress=print)
    print(json.dumps({k: v for k, v in out.items() if k in ("forecast", "delivery_risk", "stockout", "automation", "seconds")},
                     indent=1, default=str)[:4000])


if __name__ == "__main__":
    main()
