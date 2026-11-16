"""Pre-download pretrained pose weights (otherwise they download on first analysis).

    python download_models.py                 # default model (yolo26n-pose)
    python download_models.py yolo26s-pose    # one or more specific models
"""
import sys

from app.core.config import get_settings
from app.vision.constants import AVAILABLE_MODELS
from app.vision.runtime import ensure_model, torch_info

if __name__ == "__main__":
    get_settings().ensure_dirs()
    names = sys.argv[1:] or ["yolo26n-pose"]
    for n in names:
        if n.removesuffix(".pt") not in AVAILABLE_MODELS:
            sys.exit(f"Unknown model {n}. Choose from: {', '.join(AVAILABLE_MODELS)}")
        print("ok:", ensure_model(n))
    t = torch_info()
    print(f"PyTorch {t['torch']} | CUDA available: {t['cuda_available']} {t['gpus'] or ''}")
