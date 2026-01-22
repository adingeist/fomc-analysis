#!/usr/bin/env python3
"""CLI for loading backtest artifacts into the analytics database."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

from fomc_analysis.db.ingestion import load_backtest_artifacts
from fomc_analysis.db.session import resolve_database_url, session_scope


def parse_timestamp(value: str | None) -> datetime | None:
    if not value:
        return None
    text = value.strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
        return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)
    except ValueError as exc:  # pragma: no cover - argparse already validates format
        raise argparse.ArgumentTypeError(f"Invalid timestamp: {value}") from exc


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Load backtest outputs into the DB")
    parser.add_argument(
        "--backtest-json",
        type=Path,
        default=Path("results/backtest_v3/backtest_results.json"),
        help="Path to backtest_results.json",
    )
    parser.add_argument(
        "--predictions-csv",
        type=Path,
        default=None,
        help="Optional predictions.csv to enrich rows",
    )
    parser.add_argument(
        "--trades-csv",
        type=Path,
        default=None,
        help="Optional trades.csv to ingest executed trades",
    )
    parser.add_argument(
        "--dataset-slug",
        default="fomc_backtest_v3",
        help="Logical dataset slug (e.g., fomc_backtest_v3)",
    )
    parser.add_argument(
        "--dataset-type",
        default="fomc",
        help="Dataset family (e.g., fomc, earnings)",
    )
    parser.add_argument(
        "--run-id",
        default=None,
        help="Optional stable run identifier for idempotent loads",
    )
    parser.add_argument(
        "--run-timestamp",
        type=parse_timestamp,
        default=None,
        help="Override timestamp (ISO-8601)",
    )
    parser.add_argument(
        "--database-url",
        default=None,
        help="SQLAlchemy URL override (defaults to DATABASE_URL/Settings)",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    db_url = resolve_database_url(args.database_url)
    with session_scope(db_url) as session:
        dataset_run = load_backtest_artifacts(
            session,
            backtest_json=args.backtest_json,
            predictions_csv=args.predictions_csv,
            trades_csv=args.trades_csv,
            dataset_slug=args.dataset_slug,
            dataset_type=args.dataset_type,
            run_id=args.run_id,
            run_timestamp=args.run_timestamp,
        )
        print(
            f"Loaded backtest artifacts for run_id={dataset_run.run_id} into dataset {dataset_run.dataset_slug}"
        )


if __name__ == "__main__":
    main()
