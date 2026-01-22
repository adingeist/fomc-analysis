#!/usr/bin/env python3
"""CLI for loading grid search results into the analytics database."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

from fomc_analysis.db.ingestion import load_grid_search_results
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
    parser = argparse.ArgumentParser(description="Load grid search CSV into the DB")
    parser.add_argument(
        "--grid-search-csv",
        type=Path,
        default=Path("results/backtest_v3/grid_search.csv"),
        help="Path to the grid_search.csv file",
    )
    parser.add_argument("--dataset-slug", default="fomc_backtest_v3_grid_search")
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
        dataset_run = load_grid_search_results(
            session,
            grid_search_csv=args.grid_search_csv,
            dataset_slug=args.dataset_slug,
            dataset_type=args.dataset_type,
            run_id=args.run_id,
            run_timestamp=args.run_timestamp,
        )
        print(
            f"Loaded {dataset_run.dataset_slug} grid search rows for run_id={dataset_run.run_id}"
        )


if __name__ == "__main__":
    main()
