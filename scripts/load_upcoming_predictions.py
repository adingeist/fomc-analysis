#!/usr/bin/env python3
"""CLI for loading upcoming prediction exports."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

from fomc_analysis.db.ingestion import load_upcoming_predictions
from fomc_analysis.db.session import resolve_database_url, session_scope


def parse_timestamp(value: str | None) -> datetime | None:
    if not value:
        return None
    text = value.strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    parsed = datetime.fromisoformat(text)
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Load upcoming predictions CSV into the DB")
    parser.add_argument(
        "--predictions-csv",
        type=Path,
        default=Path("results/upcoming_predictions/predictions.csv"),
        help="CSV containing live predictions",
    )
    parser.add_argument("--dataset-slug", default="fomc_upcoming_predictions")
    parser.add_argument("--dataset-type", default="fomc")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--run-timestamp", type=parse_timestamp, default=None)
    parser.add_argument("--database-url", default=None)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    db_url = resolve_database_url(args.database_url)
    with session_scope(db_url) as session:
        dataset_run = load_upcoming_predictions(
            session,
            predictions_csv=args.predictions_csv,
            dataset_slug=args.dataset_slug,
            dataset_type=args.dataset_type,
            run_id=args.run_id,
            run_timestamp=args.run_timestamp,
        )
        print(
            f"Loaded upcoming predictions from {args.predictions_csv} for run_id={dataset_run.run_id}"
        )


if __name__ == "__main__":
    main()
