"""Device selection (CUDA when available, otherwise CPU) and pretrained model management."""
from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from pathlib import Path

from app.core.config import get_settings

log = logging.getLogger(__name__)
_lock = threading.Lock()


@dataclass
class DeviceChoice:
    device: str  # value passed to Ultralytics ("cpu", "cuda:0", "mps")
    half: bool
    label: str  # human readable, e.g. "NVIDIA RTX 4070 (cuda:0)"
    warning: str | None = None


def torch_info() -> dict:
    """Best-effort description of the PyTorch / accelerator setup."""
    info: dict = {"torch": None, "cuda_available": False, "cuda_version": None, "gpus": [], "mps_available": False}
    try:
        import torch

        info["torch"] = torch.__version__
        info["cuda_available"] = bool(torch.cuda.is_available())
        info["cuda_version"] = torch.version.cuda
        if info["cuda_available"]:
            info["gpus"] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
        mps = getattr(torch.backends, "mps", None)
        info["mps_available"] = bool(mps and mps.is_available())
        import os

        info["cpu_threads"] = os.cpu_count()
    except Exception as exc:  # pragma: no cover - depends on install
        info["error"] = str(exc)
    return info


def resolve_device(requested: str, half: bool) -> DeviceChoice:
    """Map the user's setting to a usable device, falling back to CPU with a warning."""
    import torch

    cuda_ok = torch.cuda.is_available()
    mps = getattr(torch.backends, "mps", None)
    mps_ok = bool(mps and mps.is_available())

    if requested == "auto":
        if cuda_ok:
            return DeviceChoice("cuda:0", half, f"{torch.cuda.get_device_name(0)} (cuda:0)")
        if mps_ok:
            return DeviceChoice("mps", False, "Apple GPU (mps)")
        return DeviceChoice("cpu", False, "CPU")

    if requested.startswith("cuda"):
        idx = int(requested.split(":")[1]) if ":" in requested else 0
        if cuda_ok and idx < torch.cuda.device_count():
            return DeviceChoice(f"cuda:{idx}", half, f"{torch.cuda.get_device_name(idx)} (cuda:{idx})")
        return DeviceChoice("cpu", False, "CPU", warning=f"Requested {requested} but CUDA is not available to PyTorch; using CPU")

    if requested == "mps":
        if mps_ok:
            return DeviceChoice("mps", False, "Apple GPU (mps)")
        return DeviceChoice("cpu", False, "CPU", warning="Requested mps but it is not available; using CPU")

    return DeviceChoice("cpu", False, "CPU")


def models_dir() -> Path:
    s = get_settings()
    d = s.models_dir or (s.data_dir / "models")
    d.mkdir(parents=True, exist_ok=True)
    return d


def model_path(name: str) -> Path:
    return models_dir() / f"{name.removesuffix('.pt')}.pt"


def downloaded_models() -> list[str]:
    return sorted(p.stem for p in models_dir().glob("*.pt"))


def ensure_model(name: str) -> Path:
    """Download pretrained weights from the official Ultralytics release assets on first use."""
    path = model_path(name)
    if path.exists() and path.stat().st_size > 0:
        return path
    with _lock:
        if path.exists() and path.stat().st_size > 0:
            return path
        from ultralytics.utils.downloads import attempt_download_asset

        log.info("Downloading pretrained model %s ...", name)
        try:
            result = Path(attempt_download_asset(str(path)))
        except Exception as exc:
            raise RuntimeError(
                f"Could not download model weights '{name}.pt' (internet required on first use): {exc}"
            ) from exc
        if not result.exists():
            raise RuntimeError(f"Model weights '{name}.pt' were not downloaded")
        if result.resolve() != path.resolve():
            result.replace(path)
        return path


def load_pose_model(name: str):
    """Return a *fresh* Ultralytics YOLO pose model (fresh so tracker state never leaks between jobs)."""
    from ultralytics import YOLO

    model = YOLO(str(ensure_model(name)), task="pose")
    if getattr(model, "task", "pose") != "pose":
        raise RuntimeError(f"{name} is not a pose model")
    return model
