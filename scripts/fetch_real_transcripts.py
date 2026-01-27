#!/usr/bin/env python3
"""
Fetch and ingest real earnings call transcripts.

This script helps build the real transcript dataset needed for feature engineering.
It supports multiple approaches:

1. SEC EDGAR: Fetches earnings dates and press releases (not full transcripts)
2. Manual ingestion: Processes transcripts placed in data/earnings/transcripts/
3. Word mention counting: Counts specific words across ingested transcripts

The reality: full earnings call transcripts are not freely available from SEC EDGAR.
They must be sourced from:
- Company investor relations pages
- Seeking Alpha, Motley Fool (may require subscription)
- Financial Modeling Prep API (free tier)
- Manual copy-paste from earnings call replays

Usage:
    # Fetch earnings dates from SEC EDGAR
    python scripts/fetch_real_transcripts.py --mode dates --ticker META

    # Ingest manually-placed transcripts
    python scripts/fetch_real_transcripts.py --mode ingest --ticker META

    # Count word mentions in ingested transcripts
    python scripts/fetch_real_transcripts.py --mode count --ticker META --words "AI,VR,metaverse"

    # Full pipeline: dates + ingest + count
    python scripts/fetch_real_transcripts.py --mode all --ticker META --words "AI,VR,metaverse"
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from earnings_analysis.fetchers.sec_edgar_transcripts import (
    SECEdgarFetcher,
    ManualTranscriptIngester,
)


def cmd_dates(args):
    """Fetch earnings dates from SEC EDGAR."""
    fetcher = SECEdgarFetcher(output_dir=args.output_dir / "sec_filings")

    print(f"Fetching earnings filing dates for {args.ticker}...")
    filings = fetcher.fetch_earnings_filings(args.ticker, args.num_quarters)

    if not filings:
        print(f"No earnings filings found for {args.ticker}")
        return

    print(f"\nFound {len(filings)} earnings filings:")
    for f in filings:
        items = ", ".join(f.items_reported)
        extras = []
        if f.has_press_release:
            extras.append("press release")
        if f.has_transcript:
            extras.append("transcript")
        extra_str = f" [{', '.join(extras)}]" if extras else ""
        print(f"  {f.filing_date}: {f.form_type} (Items: {items}){extra_str}")

    # Save filings metadata
    output_file = args.output_dir / "sec_filings" / f"{args.ticker}_filings.json"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w") as fh:
        json.dump([f.to_dict() for f in filings], fh, indent=2, default=str)
    print(f"\nSaved to {output_file}")

    # Download press releases if available
    press_releases = [f for f in filings if f.has_press_release]
    if press_releases:
        print(f"\nDownloading {len(press_releases)} press releases...")
        for filing in press_releases:
            text = fetcher.download_press_release(filing)
            if text:
                pr_dir = args.output_dir / "press_releases" / args.ticker
                pr_dir.mkdir(parents=True, exist_ok=True)
                pr_file = pr_dir / f"{args.ticker}_{filing.filing_date}.txt"
                pr_file.write_text(text)
                print(f"  Saved: {pr_file}")


def cmd_ingest(args):
    """Ingest manually-placed transcripts."""
    transcripts_dir = args.output_dir / "transcripts"
    segments_dir = args.output_dir / "segments"

    ticker_dir = transcripts_dir / args.ticker
    if not ticker_dir.exists():
        print(f"No transcript directory found at {ticker_dir}")
        print(f"\nTo use manual ingestion:")
        print(f"  1. Create directory: {ticker_dir}")
        print(f"  2. Place transcript files named: {args.ticker}_YYYY-MM-DD.txt")
        print(f"  3. Re-run this command")
        print(f"\nExpected format:")
        print(f"  {ticker_dir}/{args.ticker}_2025-01-29.txt")
        print(f"  {ticker_dir}/{args.ticker}_2025-04-30.txt")
        return

    ingester = ManualTranscriptIngester(
        transcripts_dir=transcripts_dir,
        segments_dir=segments_dir,
    )

    records = ingester.ingest_directory(args.ticker)
    print(f"\nIngested {len(records)} transcripts")

    for r in records:
        print(f"  {r.call_date}: {len(r.segments)} segments ({r.source})")


def cmd_count(args):
    """Count word mentions in ingested transcripts."""
    if not args.words:
        print("--words is required for count mode (comma-separated)")
        return

    segments_dir = args.output_dir / "segments"
    words = [w.strip() for w in args.words.split(",")]

    ingester = ManualTranscriptIngester(segments_dir=segments_dir)
    df = ingester.count_word_mentions(args.ticker, words)

    if df.empty:
        print(f"No segments found for {args.ticker}")
        print(f"Run with --mode ingest first to process transcripts")
        return

    print(f"\nWord mention matrix for {args.ticker}:")
    print(df.to_string())

    # Save
    output_file = args.output_dir / "features" / f"{args.ticker}_word_mentions.csv"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_file)
    print(f"\nSaved to {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Fetch and ingest real earnings call transcripts"
    )
    parser.add_argument(
        "--mode",
        choices=["dates", "ingest", "count", "all"],
        default="dates",
        help="Operation mode",
    )
    parser.add_argument("--ticker", type=str, required=True, help="Stock ticker")
    parser.add_argument("--num-quarters", type=int, default=8)
    parser.add_argument("--words", type=str, help="Comma-separated words to count")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/earnings"),
        help="Base output directory",
    )
    args = parser.parse_args()

    print("=" * 70)
    print("REAL EARNINGS TRANSCRIPT PIPELINE")
    print("=" * 70)
    print(f"Ticker: {args.ticker}")
    print(f"Mode: {args.mode}")

    if args.mode == "dates":
        cmd_dates(args)
    elif args.mode == "ingest":
        cmd_ingest(args)
    elif args.mode == "count":
        cmd_count(args)
    elif args.mode == "all":
        cmd_dates(args)
        print("\n" + "=" * 70)
        cmd_ingest(args)
        if args.words:
            print("\n" + "=" * 70)
            cmd_count(args)


if __name__ == "__main__":
    main()
