"""Download the public datasets into data/raw/ (run on a machine with internet access).

Kaggle datasets use kagglehub (needs a Kaggle account: ~/.kaggle/kaggle.json or KAGGLE_USERNAME/KAGGLE_KEY).
UCI Online Retail is downloaded from the UCI repository.

    python scripts/download_datasets.py            # Olist + DataCo + UCI
    python scripts/download_datasets.py --m5       # also the M5 competition files (accept the rules first)
"""
from __future__ import annotations

import argparse
import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path

RAW = Path(__file__).resolve().parents[1] / "data" / "raw"
UCI_URL = "https://archive.ics.uci.edu/static/public/352/online+retail.zip"


def copy_tree(src: Path, dst: Path) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    for p in Path(src).rglob("*"):
        if p.is_file():
            target = dst / p.name
            shutil.copy2(p, target)
            if target.suffix == ".zip":
                with zipfile.ZipFile(target) as z:
                    z.extractall(dst)
    print(f"  -> {dst}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--m5", action="store_true")
    args = ap.parse_args()
    import kagglehub
    print("Olist (olistbr/brazilian-ecommerce)")
    copy_tree(Path(kagglehub.dataset_download("olistbr/brazilian-ecommerce")), RAW / "olist")
    print("DataCo (shashwatwork/dataco-smart-supply-chain-for-big-data-analysis)")
    copy_tree(Path(kagglehub.dataset_download("shashwatwork/dataco-smart-supply-chain-for-big-data-analysis")), RAW / "dataco")
    if args.m5:
        print("M5 (competition m5-forecasting-accuracy)")
        copy_tree(Path(kagglehub.competition_download("m5-forecasting-accuracy")), RAW / "m5")
    uci = RAW / "uci"
    uci.mkdir(parents=True, exist_ok=True)
    if not list(uci.glob("*.xlsx")):
        print("UCI Online Retail")
        zp = uci / "online_retail.zip"
        urllib.request.urlretrieve(UCI_URL, zp)
        with zipfile.ZipFile(zp) as z:
            z.extractall(uci)
    print("Done. Run: python scripts/bootstrap.py")


if __name__ == "__main__":
    sys.exit(main())
