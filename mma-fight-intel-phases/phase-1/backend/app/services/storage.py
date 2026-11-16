"""Local file storage with safe, server-generated paths.

User-supplied filenames are never used to build filesystem paths. Everything
is stored under ``settings.data_dir`` and referenced in the database by a
path *relative* to it, so the data folder can be moved as a unit.
"""
from __future__ import annotations

import logging
import os
import shutil
import time
from pathlib import Path

from app.core.config import Settings, get_settings

log = logging.getLogger(__name__)


class UnsafePathError(ValueError):
    pass


class Storage:
    def __init__(self, settings: Settings | None = None):
        self.settings = settings or get_settings()
        self.root = self.settings.data_dir.resolve()

    # ---- path helpers -------------------------------------------------
    def to_relative(self, path: Path) -> str:
        return self._ensure_inside(path).relative_to(self.root).as_posix()

    def resolve(self, relative: str | None) -> Path | None:
        if not relative:
            return None
        return self._ensure_inside(self.root / relative)

    def _ensure_inside(self, path: Path) -> Path:
        resolved = Path(path).resolve()
        if resolved != self.root and self.root not in resolved.parents:
            raise UnsafePathError(f"Path escapes data directory: {path}")
        return resolved

    def video_path(self, video_id: str, ext: str) -> Path:
        return self.settings.videos_dir / f"{video_id}{ext.lower()}"

    def thumbnail_path(self, video_id: str) -> Path:
        return self.settings.thumbnails_dir / f"{video_id}.jpg"

    def preview_path(self, video_id: str) -> Path:
        return self.settings.previews_dir / f"{video_id}.mp4"

    def frames_dir(self, video_id: str) -> Path:
        return self.settings.frames_dir / video_id

    def temp_upload_path(self, token: str) -> Path:
        return self.settings.tmp_dir / f"{token}.part"

    # ---- deletion -----------------------------------------------------
    @staticmethod
    def remove_file(path: Path | None, retries: int = 5) -> bool:
        """Delete a file, retrying briefly (Windows may hold locks for a moment)."""
        if path is None or not path.exists():
            return True
        for attempt in range(retries):
            try:
                path.unlink()
                return True
            except PermissionError:
                time.sleep(0.2 * (attempt + 1))
            except FileNotFoundError:
                return True
        log.warning("Could not delete file (still locked?): %s", path)
        return False

    @staticmethod
    def remove_tree(path: Path | None, retries: int = 5) -> bool:
        if path is None or not path.exists():
            return True
        for attempt in range(retries):
            try:
                shutil.rmtree(path)
                return True
            except PermissionError:
                time.sleep(0.2 * (attempt + 1))
            except FileNotFoundError:
                return True
        log.warning("Could not delete directory: %s", path)
        return False

    def delete_video_artifacts(self, video_id: str, stored_relative: str | None) -> None:
        """Remove everything the app stored for a video."""
        try:
            self.remove_file(self.resolve(stored_relative))
        except UnsafePathError:
            log.error("Refusing to delete unsafe stored path for %s", video_id)
        self.remove_derived_artifacts(video_id)

    def remove_derived_artifacts(self, video_id: str) -> None:
        self.remove_file(self.thumbnail_path(video_id))
        self.remove_file(self.preview_path(video_id))
        self.remove_tree(self.frames_dir(video_id))

    def cleanup_temp(self) -> int:
        removed = 0
        for p in self.settings.tmp_dir.glob("*.part"):
            if self.remove_file(p, retries=1):
                removed += 1
        return removed

    def sweep_orphans(self, known_video_ids: set[str]) -> int:
        """Remove stored files whose video no longer exists (e.g. a delete that hit a Windows file lock)."""
        removed = 0
        s = self.settings
        for folder in (s.videos_dir, s.thumbnails_dir, s.previews_dir):
            for p in folder.glob("*"):
                vid = p.name.split(".", 1)[0]
                if p.is_file() and vid not in known_video_ids and self.remove_file(p, retries=1):
                    removed += 1
        for d in s.frames_dir.glob("*"):
            if d.is_dir() and d.name not in known_video_ids and self.remove_tree(d, retries=1):
                removed += 1
        return removed

    # ---- stats ----------------------------------------------------------
    @staticmethod
    def dir_size(path: Path) -> int:
        total = 0
        if not path.exists():
            return 0
        for dirpath, _dirs, files in os.walk(path):
            for f in files:
                try:
                    total += os.path.getsize(os.path.join(dirpath, f))
                except OSError:
                    pass
        return total

    def usage(self) -> dict[str, int]:
        s = self.settings
        return {
            "videos": self.dir_size(s.videos_dir),
            "thumbnails": self.dir_size(s.thumbnails_dir),
            "frames": self.dir_size(s.frames_dir),
            "previews": self.dir_size(s.previews_dir),
        }
