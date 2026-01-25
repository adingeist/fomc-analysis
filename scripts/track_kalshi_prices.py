#!/usr/bin/env python
"""
Track daily Kalshi earnings contract prices.

This script records current market prices for active contracts.
Run daily via cron or manually to build price history.

Usage:
    # Track prices for META
    python scripts/track_kalshi_prices.py META

    # Track prices for multiple tickers
    python scripts/track_kalshi_prices.py META TSLA NVDA

    # Track all supported tickers
    python scripts/track_kalshi_prices.py --all

    # Show price history
    python scripts/track_kalshi_prices.py META --history

    # Analyze price evolution
    python scripts/track_kalshi_prices.py META --analyze --word ai

    # Export to specific date (for backfilling)
    python scripts/track_kalshi_prices.py META --date 2026-01-24
"""

import argparse
import sys
from datetime import date
from pathlib import Path


# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from earnings_analysis.fetchers.kalshi_price_tracker import (
    KalshiPriceTracker,
    PriceEvolutionAnalyzer,
    record_daily_prices,
)


# Default tickers to track
DEFAULT_TICKERS = ["META", "TSLA", "NVDA", "AMZN", "AAPL", "MSFT"]


def main():
    parser = argparse.ArgumentParser(
        description="Track Kalshi earnings contract prices",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "tickers",
        nargs="*",
        help="Tickers to track (e.g., META TSLA NVDA)",
    )

    parser.add_argument(
        "--all",
        action="store_true",
        help="Track all default tickers",
    )

    parser.add_argument(
        "--date",
        type=str,
        default=None,
        help="Snapshot date (YYYY-MM-DD, default: today)",
    )

    parser.add_argument(
        "--history",
        action="store_true",
        help="Show price history instead of recording",
    )

    parser.add_argument(
        "--analyze",
        action="store_true",
        help="Analyze price evolution",
    )

    parser.add_argument(
        "--word",
        type=str,
        default=None,
        help="Specific word to analyze (for --history or --analyze)",
    )

    parser.add_argument(
        "--call-date",
        type=str,
        default=None,
        help="Filter to specific call date (YYYY-MM-DD)",
    )

    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/kalshi_prices",
        help="Directory for price data (default: data/kalshi_prices)",
    )

    parser.add_argument(
        "--optimal-window",
        action="store_true",
        help="Analyze optimal trading window (requires --word)",
    )

    args = parser.parse_args()

    # Determine tickers
    if args.all:
        tickers = DEFAULT_TICKERS
    elif args.tickers:
        tickers = [t.upper() for t in args.tickers]
    else:
        parser.print_help()
        print("\nError: Please specify tickers or use --all")
        sys.exit(1)

    data_dir = Path(args.data_dir)

    # Initialize tracker
    tracker = KalshiPriceTracker(data_dir=data_dir)

    # Handle different modes
    if args.history:
        show_history(tracker, tickers, args.word, args.call_date)
    elif args.analyze:
        analyze_prices(tracker, tickers, args.word, args.call_date)
    elif args.optimal_window:
        if not args.word:
            print("Error: --optimal-window requires --word")
            sys.exit(1)
        analyze_optimal_window(tracker, tickers, args.word)
    else:
        record_prices(tracker, tickers, args.date)


def record_prices(
    tracker: KalshiPriceTracker,
    tickers: list,
    snapshot_date: str = None,
):
    """Record current prices for tickers."""
    print("=" * 60)
    print("Recording Kalshi Earnings Contract Prices")
    print("=" * 60)

    snapshot_date = snapshot_date or date.today().isoformat()
    print(f"\nSnapshot date: {snapshot_date}")
    print(f"Tickers: {', '.join(tickers)}")

    total_snapshots = 0

    for ticker in tickers:
        print(f"\n--- {ticker} ---")
        try:
            snapshots = tracker.record_snapshot(ticker, snapshot_date=snapshot_date)
            total_snapshots += len(snapshots)

            if snapshots:
                print(f"  Recorded {len(snapshots)} contracts:")
                # Show sample
                for s in snapshots[:5]:
                    print(f"    {s.word}: ${s.last_price:.2f} "
                          f"(bid: ${s.yes_bid or 0:.2f}, ask: ${s.yes_ask or 0:.2f})")
                if len(snapshots) > 5:
                    print(f"    ... and {len(snapshots) - 5} more")
        except Exception as e:
            print(f"  Error: {e}")

    print(f"\n{'=' * 60}")
    print(f"Total snapshots recorded: {total_snapshots}")
    print(f"Data saved to: {tracker.data_dir}")


def show_history(
    tracker: KalshiPriceTracker,
    tickers: list,
    word: str = None,
    call_date: str = None,
):
    """Show price history for tickers."""
    print("=" * 60)
    print("Price History")
    print("=" * 60)

    for ticker in tickers:
        print(f"\n--- {ticker} ---")

        df = tracker.get_price_history(ticker, word=word, call_date=call_date)

        if df.empty:
            print("  No price history found")
            continue

        # Show summary by word
        summary = df.groupby("word").agg({
            "snapshot_date": ["min", "max", "count"],
            "last_price": ["min", "max", "mean"],
        })

        print(f"\n  Found {len(df)} snapshots for {df['word'].nunique()} words:")
        print()

        for word_name in df["word"].unique():
            word_df = df[df["word"] == word_name]
            print(f"  {word_name}:")
            print(f"    Snapshots: {len(word_df)}")
            print(f"    Date range: {word_df['snapshot_date'].min().date()} to "
                  f"{word_df['snapshot_date'].max().date()}")
            print(f"    Price range: ${word_df['last_price'].min():.2f} - "
                  f"${word_df['last_price'].max():.2f}")
            print(f"    Current: ${word_df.iloc[-1]['last_price']:.2f}")
            print()


def analyze_prices(
    tracker: KalshiPriceTracker,
    tickers: list,
    word: str = None,
    call_date: str = None,
):
    """Analyze price evolution."""
    print("=" * 60)
    print("Price Evolution Analysis")
    print("=" * 60)

    analyzer = PriceEvolutionAnalyzer(tracker)

    for ticker in tickers:
        print(f"\n--- {ticker} ---")

        if word:
            words = [word.lower()]
        else:
            # Get all words from history
            df = tracker.get_price_history(ticker, call_date=call_date)
            if df.empty:
                print("  No price history found")
                continue
            words = df["word"].unique()

        for w in words[:5]:  # Limit to 5 words for readability
            print(f"\n  Word: {w}")

            evolution_df = analyzer.analyze_price_evolution(
                ticker, w, call_date=call_date
            )

            if evolution_df.empty:
                print("    No data")
                continue

            # Show evolution summary
            print(f"    Snapshots: {len(evolution_df)}")
            print(f"    Starting price: ${evolution_df.iloc[0]['last_price']:.2f}")
            print(f"    Current price: ${evolution_df.iloc[-1]['last_price']:.2f}")

            total_change = evolution_df.iloc[-1]["price_change_cumulative"]
            if total_change is not None:
                direction = "up" if total_change > 0 else "down"
                print(f"    Total change: {total_change:+.2%} ({direction})")

            volatility = evolution_df["volatility_5d"].iloc[-1]
            if volatility is not None and not pd.isna(volatility):
                print(f"    5-day volatility: {volatility:.4f}")


def analyze_optimal_window(
    tracker: KalshiPriceTracker,
    tickers: list,
    word: str,
):
    """Analyze optimal trading window."""
    print("=" * 60)
    print("Optimal Trading Window Analysis")
    print("=" * 60)

    analyzer = PriceEvolutionAnalyzer(tracker)

    for ticker in tickers:
        print(f"\n--- {ticker} - {word} ---")

        result = analyzer.find_optimal_trading_window(ticker, word)

        if "error" in result:
            print(f"  Error: {result['error']}")
            print(f"  Snapshots found: {result.get('snapshots_found', 0)}")
            continue

        print(f"  Total snapshots: {result['total_snapshots']}")
        print(f"  Recommended window: {result['recommended_window']}")

        if result.get("avg_spread_by_bucket"):
            print("\n  Average spread by time bucket:")
            for bucket, spread in result["avg_spread_by_bucket"].items():
                print(f"    {bucket}: {spread:.4f}")

        print("\n  Notes:")
        for note in result.get("analysis_notes", []):
            print(f"    - {note}")


# Import pandas for the analyze function
import pandas as pd


if __name__ == "__main__":
    main()
