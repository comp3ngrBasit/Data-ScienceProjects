"""Copy dataset files you already downloaded into data/raw/ (zips are extracted).

Example (Windows PowerShell):
    python scripts/prepare_data.py `
      --uci "C:/Users/me/Downloads/online+retail.zip" `
      --dataco "C:/Users/me/Downloads/DataCoSupplyChainDataset.csv.zip" `
      --olist "C:/Users/me/Downloads/brazilian-ecommerce.zip" `
      --m5 "C:/Users/me/Downloads/calendar.csv"
Each option accepts a .zip, a single file or a folder (e.g. the path printed by kagglehub).
"""
from __future__ import annotations

import argparse
import shutil
import zipfile
from pathlib import Path

RAW = Path(__file__).resolve().parents[1] / "data" / "raw"


def place(src: str, name: str) -> None:
    src_p, dst = Path(src).expanduser(), RAW / name
    dst.mkdir(parents=True, exist_ok=True)
    if not src_p.exists():
        raise SystemExit(f"{name}: {src_p} not found")
    files = [p for p in src_p.rglob("*") if p.is_file()] if src_p.is_dir() else [src_p]
    for f in files:
        if f.suffix.lower() == ".zip":
            with zipfile.ZipFile(f) as z:
                z.extractall(dst)
        else:
            shutil.copy2(f, dst / f.name)
    print(f"{name}: {sorted(p.name for p in dst.iterdir())}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for k in ("uci", "dataco", "olist", "m5"):
        ap.add_argument(f"--{k}")
    a = ap.parse_args()
    for k in ("uci", "dataco", "olist", "m5"):
        if getattr(a, k):
            place(getattr(a, k), k)
    print("Next: python scripts/bootstrap.py")


if __name__ == "__main__":
    main()
