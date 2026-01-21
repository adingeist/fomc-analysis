#!/usr/bin/env python3
"""
Create demo data for testing backtest v3.

This script creates mock resolved contracts based on actual
transcript data, allowing you to test the full pipeline without
waiting for real Kalshi markets to resolve.
"""

import json
import random
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import yaml


def load_segments(segments_dir: Path):
    """Load all segment files and get meeting dates."""
    meeting_dates = []

    for segment_file in sorted(segments_dir.glob("*.jsonl")):
        date_str = segment_file.stem
        try:
            date = datetime.strptime(date_str, "%Y%m%d")
            meeting_dates.append(date)
        except ValueError:
            continue

    return sorted(meeting_dates)


def count_word_mentions(segments_dir: Path, word: str) -> dict:
    """Count how many times a word is mentioned in each transcript."""
    from fomc_analysis.parsing.speaker_segmenter import load_segments_jsonl

    counts = {}

    for segment_file in sorted(segments_dir.glob("*.jsonl")):
        date_str = segment_file.stem

        try:
            segments = load_segments_jsonl(segment_file)
        except Exception:
            continue

        # Get Powell's text
        powell_text = " ".join(
            seg.text.lower() for seg in segments if seg.role == "powell"
        )

        # Count mentions (simple word boundary search)
        count = powell_text.count(word.lower())
        counts[date_str] = count

    return counts


def create_demo_contract_words(segments_dir: Path, num_contracts: int = 10) -> list:
    """Create demo contract words with mock market data."""

    # Common words to track
    words = [
        ("Inflation", 40),
        ("Unemployment", 1),
        ("Uncertainty", 1),
        ("Recession", 1),
        ("Growth", 20),
        ("Data", 50),
        ("Forecast", 1),
        ("Expectation", 1),
        ("Volatility", 1),
        ("Good Afternoon", 1),
    ][:num_contracts]

    meeting_dates = load_segments(segments_dir)

    if not meeting_dates:
        raise ValueError(f"No segments found in {segments_dir}")

    print(f"Found {len(meeting_dates)} meetings")
    print(f"Creating {num_contracts} demo contracts")

    contract_words = []

    for word, threshold in words:
        print(f"\nProcessing: {word} ({threshold}+)")

        # Count actual mentions
        mentions = count_word_mentions(segments_dir, word)

        # Create markets for each meeting date
        markets = []

        for meeting_date in meeting_dates[-15:]:  # Use last 15 meetings
            date_str = meeting_date.strftime("%Y%m%d")

            if date_str not in mentions:
                continue

            actual_count = mentions[date_str]

            # Determine outcome based on actual mentions
            outcome = "yes" if actual_count >= threshold else "no"

            # Generate realistic close price
            if outcome == "yes":
                # YES market: high price (70-99 cents)
                close_price = random.uniform(0.70, 0.99)
            else:
                # NO market: low price (1-30 cents)
                close_price = random.uniform(0.01, 0.30)

            # Create mock ticker
            ticker = f"KXFED-{word.upper().replace(' ', '')}-{meeting_date.strftime('%d%b').upper()}"

            market = {
                "ticker": ticker,
                "subtitle": f"Will Chair Powell say '{word}' {threshold}+ times?",
                "open_time": (meeting_date - timedelta(days=30)).isoformat(),
                "close_date": meeting_date.strftime("%Y-%m-%d"),
                "expiration_date": meeting_date.strftime("%Y-%m-%d"),
                "status": "resolved",
                "result": outcome,
                "close_price": close_price,
                "actual_count": actual_count,
            }

            markets.append(market)

        print(f"  Created {len(markets)} resolved markets")

        # Display name
        if threshold > 1:
            display_word = f"{word} ({threshold}+)"
        else:
            display_word = word

        contract_words.append({
            "word": display_word,
            "threshold": threshold,
            "base_phrases": [word.lower()],
            "markets": markets,
        })

    return contract_words


def main():
    """Create demo data."""
    print("=" * 80)
    print("CREATING DEMO DATA FOR BACKTEST V3")
    print("=" * 80)

    segments_dir = Path("data/segments")
    output_dir = Path("data/kalshi_analysis")

    if not segments_dir.exists():
        print(f"\n✗ Error: Segments directory not found: {segments_dir}")
        print("Run the pipeline first to fetch and parse transcripts:")
        print("  python run_e2e_backtest.py --skip-analyze")
        return 1

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate demo contract words
    print("\nGenerating demo contract words...")
    contract_words = create_demo_contract_words(segments_dir, num_contracts=10)

    # Save to JSON
    output_file = output_dir / "contract_words.json"
    output_file.write_text(json.dumps(contract_words, indent=2))

    print(f"\n✓ Demo data saved to: {output_file}")

    # Create summary
    print("\n" + "=" * 80)
    print("DEMO DATA SUMMARY")
    print("=" * 80)

    total_markets = sum(len(contract["markets"]) for contract in contract_words)
    print(f"\nContracts: {len(contract_words)}")
    print(f"Total resolved markets: {total_markets}")

    print("\nContract breakdown:")
    for contract in contract_words:
        resolved_markets = [m for m in contract["markets"] if m["status"] == "resolved"]
        yes_outcomes = sum(1 for m in resolved_markets if m["result"] == "yes")
        no_outcomes = len(resolved_markets) - yes_outcomes

        print(f"  {contract['word']}: {len(resolved_markets)} markets "
              f"({yes_outcomes} YES, {no_outcomes} NO)")

    print("\n" + "=" * 80)
    print("✓ DEMO DATA CREATED SUCCESSFULLY")
    print("=" * 80)
    print("\nYou can now run the backtest with:")
    print("  python run_e2e_backtest.py --skip-fetch --skip-parse --skip-analyze")
    print("\nOr run manually:")
    print("  fomc backtest-v3 \\")
    print("    --contract-words data/kalshi_analysis/contract_words.json \\")
    print("    --output results/backtest_v3")

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
