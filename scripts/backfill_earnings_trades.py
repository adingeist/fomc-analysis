#!/usr/bin/env python3
"""
Backfill trade and market data for all earnings mention contracts.

Usage:
    python scripts/backfill_earnings_trades.py [--companies META,TSLA] [--trades] [--snapshot]

Options:
    --companies  Comma-separated list of tickers (default: all 8)
    --trades     Fetch trade-level data (slower, paginated)
    --snapshot   Take a market price snapshot (fast)
    --min-volume Minimum volume to fetch trades for (default: 10)

Examples:
    # Snapshot all companies (fast)
    python scripts/backfill_earnings_trades.py --snapshot

    # Full trade backfill for META and TSLA
    python scripts/backfill_earnings_trades.py --trades --companies META,TSLA

    # Everything for all companies
    python scripts/backfill_earnings_trades.py --trades --snapshot
"""

import argparse
import json
import sys
from pathlib import Path

# Ensure project is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from earnings_analysis.microstructure.trade_storage import ParquetStorage
from earnings_analysis.microstructure.trade_fetcher import (
    EarningsTradesFetcher,
    EARNINGS_TICKERS,
)


def main():
    parser = argparse.ArgumentParser(
        description="Backfill Kalshi earnings trade data"
    )
    parser.add_argument(
        "--companies",
        type=str,
        default=None,
        help="Comma-separated company tickers (default: all)",
    )
    parser.add_argument(
        "--trades",
        action="store_true",
        help="Fetch trade-level data (paginated)",
    )
    parser.add_argument(
        "--snapshot",
        action="store_true",
        help="Take market price snapshot",
    )
    parser.add_argument(
        "--min-volume",
        type=int,
        default=10,
        help="Min volume to fetch trades (default: 10)",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/microstructure",
        help="Data directory (default: data/microstructure)",
    )

    args = parser.parse_args()

    # Parse companies
    companies = None
    if args.companies:
        companies = [c.strip().upper() for c in args.companies.split(",")]

    # Default: snapshot if nothing specified
    if not args.trades and not args.snapshot:
        args.snapshot = True

    storage = ParquetStorage(data_dir=Path(args.data_dir))
    fetcher = EarningsTradesFetcher(storage=storage)

    if args.snapshot:
        print("\n=== Taking market price snapshot ===")
        count = fetcher.backfill_market_snapshots(companies=companies)
        print(f"Snapshot complete: {count} records saved")

    if args.trades:
        print("\n=== Backfilling trade data ===")
        results = fetcher.backfill_earnings_trades(
            companies=companies,
            min_volume=args.min_volume,
        )
        print("\n=== Trade backfill summary ===")
        for ticker, count in results.items():
            print(f"  {ticker}: {count} new trades")

    # Print data summary
    summary = storage.data_summary()
    print(f"\n=== Data summary ===")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
