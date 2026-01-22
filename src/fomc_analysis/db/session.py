"""Database session management utilities."""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Generator

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.engine.url import make_url
from sqlalchemy.orm import Session, sessionmaker

from fomc_analysis.config import settings

DEFAULT_SQLITE_URL = "sqlite:///data/fomc_analysis.db"

_ENGINES: dict[str, Engine] = {}
_SESSION_FACTORIES: dict[str, sessionmaker] = {}


def resolve_database_url(explicit_url: str | None = None) -> str:
    """Return the configured database URL, falling back to SQLite."""

    configured_url = explicit_url or settings.database_url or DEFAULT_SQLITE_URL
    return configured_url


def _ensure_sqlite_directory(database_url: str) -> None:
    """Create parent folders for SQLite files if they do not exist."""

    try:
        url = make_url(database_url)
    except Exception:
        return

    if not url.drivername.startswith("sqlite"):
        return

    database_path = url.database
    if not database_path or database_path == ":memory:":
        return

    path = Path(database_path)
    if not path.is_absolute():
        path = Path.cwd() / path
    path.parent.mkdir(parents=True, exist_ok=True)


def get_engine(database_url: str | None = None) -> Engine:
    """Return (or create) a SQLAlchemy engine for the provided URL."""

    resolved_url = resolve_database_url(database_url)
    if resolved_url not in _ENGINES:
        _ensure_sqlite_directory(resolved_url)
        _ENGINES[resolved_url] = create_engine(resolved_url, future=True)
    return _ENGINES[resolved_url]


def get_session_factory(database_url: str | None = None) -> sessionmaker:
    """Return a cached session factory for the provided URL."""

    resolved_url = resolve_database_url(database_url)
    if resolved_url not in _SESSION_FACTORIES:
        engine = get_engine(resolved_url)
        _SESSION_FACTORIES[resolved_url] = sessionmaker(bind=engine, expire_on_commit=False)
    return _SESSION_FACTORIES[resolved_url]


@contextmanager
def session_scope(database_url: str | None = None) -> Generator[Session, None, None]:
    """Provide a transactional scope for DB work."""

    session_factory = get_session_factory(database_url)
    session = session_factory()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()

