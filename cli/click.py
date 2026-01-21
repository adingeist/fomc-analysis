#!/usr/bin/env python3
"""
Standalone CLI script for fetching FOMC press conference transcript PDFs.

This script provides a command-line interface that uses the fomc_analysis package.
It can be run directly: python -m cli.click
"""

from __future__ import annotations

import time
from pathlib import Path

import click

# Import from the package
from fomc_analysis.fetcher import fetch_transcripts


@click.command()
@click.option("--start-year", type=int, default=2011, show_default=True)
@click.option("--end-year", type=int, default=time.gmtime().tm_year, show_default=True)
@click.option(
    "--out-dir",
    type=click.Path(path_type=Path),
    default=Path("data/transcripts_pdf"),
    show_default=True,
)
@click.option(
    "--index-csv",
    type=click.Path(path_type=Path),
    default=Path("data/pressconf_index.csv"),
    show_default=True,
)
@click.option(
    "--workers",
    type=int,
    default=8,
    show_default=True,
    help="Concurrent download workers",
)
@click.option(
    "--sleep",
    type=float,
    default=0.0,
    show_default=True,
    help="Sleep between year-page fetches (seconds)",
)
@click.option("--overwrite", is_flag=True, help="Overwrite existing PDFs")
@click.option("--dry-run", is_flag=True, help="Only print what would be downloaded")
def main(
    start_year: int,
    end_year: int,
    out_dir: Path,
    index_csv: Path,
    workers: int,
    sleep: float,
    overwrite: bool,
    dry_run: bool,
):
    """
    Fetch FOMC press conference transcript PDFs by scraping Fed historical year pages.
    """
    click.echo(f"Collecting meeting pages for years {start_year}..{end_year} ...")

    fetch_transcripts(
        start_year=start_year,
        end_year=end_year,
        out_dir=out_dir,
        index_csv=index_csv,
        workers=workers,
        sleep=sleep,
        overwrite=overwrite,
        dry_run=dry_run,
    )

    click.echo(f"Done. Index CSV: {index_csv}")


if __name__ == "__main__":
    main()
