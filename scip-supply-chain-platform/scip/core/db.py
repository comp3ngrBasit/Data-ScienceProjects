"""Database engine, sessions and fast DataFrame I/O."""
from __future__ import annotations

import io
from contextlib import contextmanager
from functools import lru_cache

import pandas as pd
from sqlalchemy import create_engine, delete, select, text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from scip.core.config import get_settings
from scip.core.logging import get_logger
from scip.core.models import AppMetadata, Base

log = get_logger("db")


@lru_cache
def get_engine(url: str | None = None) -> Engine:
    url = url or get_settings().database_url
    kwargs = {"pool_pre_ping": True, "future": True}
    if url.startswith("sqlite"):
        kwargs["connect_args"] = {"check_same_thread": False}
    return create_engine(url, **kwargs)


def init_db(engine: Engine | None = None) -> None:
    engine = engine or get_engine()
    Base.metadata.create_all(engine)
    log.info("Schema ensured on %s", engine.url.render_as_string(hide_password=True))


def reset_db(engine: Engine | None = None) -> None:
    engine = engine or get_engine()
    Base.metadata.drop_all(engine)
    Base.metadata.create_all(engine)


@contextmanager
def session_scope(engine: Engine | None = None):
    factory = sessionmaker(bind=engine or get_engine(), expire_on_commit=False)
    session: Session = factory()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def read_sql(sql: str, params: dict | None = None, engine: Engine | None = None,
             parse_dates: list[str] | None = None) -> pd.DataFrame:
    with (engine or get_engine()).connect() as conn:
        return pd.read_sql_query(text(sql), conn, params=params or {}, parse_dates=parse_dates)


def read_table(table: str, engine: Engine | None = None, parse_dates: list[str] | None = None) -> pd.DataFrame:
    return read_sql(f"SELECT * FROM {table}", engine=engine, parse_dates=parse_dates)


def write_frame(df: pd.DataFrame, table: str, engine: Engine | None = None, replace_where: str | None = None,
                truncate: bool = False) -> int:
    """Append a DataFrame to an existing table. Uses PostgreSQL COPY when available (fast),
    otherwise batched inserts. `truncate` empties the table first; `replace_where` deletes matching rows."""
    engine = engine or get_engine()
    if df is None or df.empty:
        return 0
    cols = [c.name for c in Base.metadata.tables[table].columns]
    df = df[[c for c in df.columns if c in cols]]
    with engine.begin() as conn:
        if truncate:
            conn.execute(Base.metadata.tables[table].delete())
        elif replace_where:
            conn.execute(text(f"DELETE FROM {table} WHERE {replace_where}"))
        if engine.dialect.name == "postgresql":
            buf = io.StringIO()
            df.to_csv(buf, index=False, header=False, na_rep="\\N")
            buf.seek(0)
            raw = conn.connection.dbapi_connection
            with raw.cursor() as cur:
                cur.copy_expert(f"COPY {table} ({','.join(df.columns)}) FROM STDIN WITH (FORMAT csv, NULL '\\N')", buf)
        else:
            df.to_sql(table, conn, if_exists="append", index=False, chunksize=2000, method="multi")
    return len(df)


def get_meta(key: str, default: str | None = None, engine: Engine | None = None) -> str | None:
    with session_scope(engine) as s:
        row = s.get(AppMetadata, key)
        return row.value if row else default


def set_meta(key: str, value, engine: Engine | None = None) -> None:
    with session_scope(engine) as s:
        row = s.get(AppMetadata, key)
        if row:
            row.value = str(value)
        else:
            s.add(AppMetadata(key=key, value=str(value)))


def table_counts(engine: Engine | None = None) -> dict[str, int]:
    out = {}
    with (engine or get_engine()).connect() as conn:
        for t in Base.metadata.sorted_tables:
            out[t.name] = conn.execute(text(f"SELECT COUNT(*) FROM {t.name}")).scalar_one()
    return out
