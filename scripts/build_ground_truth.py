#!/usr/bin/env python3
"""
Build ground truth dataset from finalized Kalshi earnings mention contracts.

Fetches all settled/finalized contracts from the Kalshi API, extracts
outcomes (which words were mentioned in which earnings calls), and saves
a structured dataset for real backtesting.

Usage:
    python scripts/build_ground_truth.py
    python scripts/build_ground_truth.py --output-dir data/ground_truth
    python scripts/build_ground_truth.py --include-active
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from earnings_analysis.ground_truth import (
    fetch_ground_truth,
    save_ground_truth,
    build_backtest_dataframes,
)


def main():
    parser = argparse.ArgumentParser(
        description="Build ground truth dataset from finalized Kalshi contracts"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/ground_truth"),
        help="Output directory for ground truth files",
    )
    parser.add_argument(
        "--include-active",
        action="store_true",
        help="Also fetch active contracts for current price snapshots",
    )
    args = parser.parse_args()

    print("=" * 70)
    print("BUILDING GROUND TRUTH DATASET FROM KALSHI FINALIZED CONTRACTS")
    print("=" * 70)

    # Step 1: Fetch ground truth
    print("\n[1/3] Fetching settled contracts from Kalshi API...")
    dataset = fetch_ground_truth(include_active=args.include_active)

    if not dataset.contracts:
        print("\nNo settled contracts found. This could mean:")
        print("  - No earnings mention contracts have settled yet")
        print("  - API credentials are not configured")
        print("  - Series tickers have changed")
        print("\nTip: Run scripts/explore_kalshi_earnings_contracts.py to check available contracts")
        return

    # Step 2: Save dataset
    print(f"\n[2/3] Saving ground truth dataset to {args.output_dir}...")
    paths = save_ground_truth(dataset, args.output_dir)

    # Step 3: Summary and validation
    print(f"\n[3/3] Dataset Summary")
    print("-" * 50)
    summary = dataset.summary()
    print(f"  Total settled contracts: {summary['total_contracts']}")
    print(f"  Companies: {', '.join(summary['companies'])}")
    print(f"  Event dates: {summary['num_event_dates']}")
    print(f"  YES outcomes: {summary['yes_outcomes']} ({summary['yes_rate']:.1%})")
    print(f"  NO outcomes: {summary['no_outcomes']}")
    print(f"  Unique words tracked: {len(summary['unique_words'])}")

    # Show per-company breakdown
    print(f"\n  Per-company breakdown:")
    for company in summary["companies"]:
        company_data = dataset.for_ticker(company)
        company_dates = set(c.event_date for c in company_data.contracts if c.event_date)
        company_words = set(c.word for c in company_data.contracts)
        company_yes = sum(1 for c in company_data.contracts if c.outcome == 1)
        print(
            f"    {company}: {len(company_data.contracts)} contracts, "
            f"{len(company_dates)} dates, {len(company_words)} words, "
            f"{company_yes} YES"
        )

    # Build and show backtest DataFrames for each company
    print(f"\n  Backtest DataFrames preview:")
    for company in summary["companies"]:
        features, outcomes, prices = build_backtest_dataframes(dataset, company)
        if not outcomes.empty:
            print(f"\n    {company}:")
            print(f"      Dates: {len(outcomes)} events")
            print(f"      Words: {list(outcomes.columns)[:5]}{'...' if len(outcomes.columns) > 5 else ''}")
            print(f"      Date range: {outcomes.index.min()} to {outcomes.index.max()}")
            print(f"      Outcomes shape: {outcomes.shape}")

    print(f"\n  Files saved:")
    for name, path in paths.items():
        print(f"    {name}: {path}")

    print("\nGround truth dataset built successfully.")
    print("Next step: Run scripts/run_real_backtest.py to backtest with this data.")


if __name__ == "__main__":
    main()
