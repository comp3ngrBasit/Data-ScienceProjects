"""SQLAlchemy engine/session management (SQLite, thread-safe for the worker)."""
from __future__ import annotations

from collections.abc import Generator
from contextlib import contextmanager

from sqlalchemy import create_engine, event
from sqlalchemy.engine import Engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker


class Base(DeclarativeBase):
    pass


_engine: Engine | None = None
_SessionLocal: sessionmaker[Session] | None = None


def init_engine(database_url: str) -> Engine:
    """Create (or recreate) the global engine. Called on app startup and in tests."""
    global _engine, _SessionLocal
    if _engine is not None:
        _engine.dispose()

    connect_args = {}
    if database_url.startswith("sqlite"):
        connect_args = {"check_same_thread": False, "timeout": 30}

    engine = create_engine(database_url, connect_args=connect_args, future=True)

    if database_url.startswith("sqlite"):

        @event.listens_for(engine, "connect")
        def _sqlite_pragmas(dbapi_conn, _record):  # pragma: no cover - trivial
            cur = dbapi_conn.cursor()
            cur.execute("PRAGMA foreign_keys=ON")
            cur.execute("PRAGMA journal_mode=WAL")
            cur.execute("PRAGMA synchronous=NORMAL")
            cur.close()

    _engine = engine
    _SessionLocal = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)
    return engine


def get_engine() -> Engine:
    if _engine is None:
        raise RuntimeError("Database engine not initialised")
    return _engine


def create_all() -> None:
    from app.db import models  # noqa: F401  (register models)

    Base.metadata.create_all(get_engine())


def get_session_factory() -> sessionmaker[Session]:
    if _SessionLocal is None:
        raise RuntimeError("Database engine not initialised")
    return _SessionLocal


def get_db() -> Generator[Session, None, None]:
    """FastAPI dependency."""
    db = get_session_factory()()
    try:
        yield db
    finally:
        db.close()


@contextmanager
def session_scope() -> Generator[Session, None, None]:
    """Transactional scope for background work."""
    db = get_session_factory()()
    try:
        yield db
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()
