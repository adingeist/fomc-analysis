"""Alembic configuration for the analytics database."""

from __future__ import annotations

from logging.config import fileConfig

from alembic import context
from sqlalchemy import engine_from_config, pool

from fomc_analysis.config import settings
from fomc_analysis.db.base import Base

# Import ORM models so that metadata is populated for autogenerate / migrations
from fomc_analysis.db import models  # noqa: F401  pylint: disable=unused-import

config = context.config
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

def get_url() -> str:
    return settings.database_url

config.set_main_option("sqlalchemy.url", get_url())

target_metadata = Base.metadata

def run_migrations_offline() -> None:
    """Run migrations without an active DB connection."""

    context.configure(
        url=get_url(),
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Run migrations against a live database connection."""

    connectable = engine_from_config(
        config.get_section(config.config_ini_section),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    with connectable.connect() as connection:
        context.configure(connection=connection, target_metadata=target_metadata)

        with context.begin_transaction():
            context.run_migrations()


def run_migrations() -> None:
    if context.is_offline_mode():
        run_migrations_offline()
    else:
        run_migrations_online()


run_migrations()
