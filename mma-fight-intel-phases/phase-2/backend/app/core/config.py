"""Application configuration.

Static settings come from environment variables (prefix ``MMA_``) or a
``backend/.env`` file. Runtime-adjustable processing settings (frame sampling
etc.) live in the database and are managed by ``services.settings_service``.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

BACKEND_DIR = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="MMA_",
        env_file=str(BACKEND_DIR / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_name: str = "MMA Fight Intelligence API"
    app_version: str = "1.0.0-phase1"

    # Storage
    data_dir: Path = BACKEND_DIR / "data"
    database_url: str | None = None  # defaults to sqlite file inside data_dir

    # HTTP
    cors_origins: list[str] = [
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:4173",
        "http://127.0.0.1:4173",
    ]

    # Uploads
    allowed_extensions: list[str] = [".mp4", ".mov", ".avi"]
    hard_max_upload_size_mb: int = 20480  # absolute ceiling; runtime limit is in settings
    upload_chunk_size: int = 1024 * 1024

    # Background processing
    worker_threads: int = 1
    preview_transcode_timeout_seconds: int = 60 * 60

    # External tools (optional explicit paths; auto-detected otherwise)
    ffmpeg_path: str | None = None
    ffprobe_path: str | None = None

    log_level: str = "INFO"

    # Phase 2: where pretrained model weights are cached (default: <data_dir>/models)
    models_dir: Path | None = None

    # ---- derived paths -------------------------------------------------
    @property
    def videos_dir(self) -> Path:
        return self.data_dir / "videos"

    @property
    def thumbnails_dir(self) -> Path:
        return self.data_dir / "thumbnails"

    @property
    def frames_dir(self) -> Path:
        return self.data_dir / "frames"

    @property
    def previews_dir(self) -> Path:
        return self.data_dir / "previews"

    @property
    def tmp_dir(self) -> Path:
        return self.data_dir / "tmp"

    @property
    def logs_dir(self) -> Path:
        return self.data_dir / "logs"

    @property
    def resolved_database_url(self) -> str:
        if self.database_url:
            return self.database_url
        return f"sqlite:///{(self.data_dir / 'mma_intel.db').as_posix()}"

    def ensure_dirs(self) -> None:
        for d in (
            self.data_dir,
            self.videos_dir,
            self.thumbnails_dir,
            self.frames_dir,
            self.previews_dir,
            self.tmp_dir,
            self.logs_dir,
        ):
            d.mkdir(parents=True, exist_ok=True)


@lru_cache
def get_settings() -> Settings:
    return Settings()
