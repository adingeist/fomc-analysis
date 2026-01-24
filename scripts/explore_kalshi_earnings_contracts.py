"""
Explore Kalshi earnings mention contracts.

This script checks what earnings call mention contracts are available on Kalshi.
"""

import asyncio
from pathlib import Path
import json

from fomc_analysis.kalshi_client_factory import get_kalshi_client


async def explore_earnings_contracts():
    """Explore available earnings mention contracts on Kalshi."""

    # Create client
    client = get_kalshi_client()

    print("=" * 60)
    print("Exploring Kalshi Earnings Mention Contracts")
    print("=" * 60)

    # Search for earnings-related series
    earnings_series_patterns = [
        "EARNINGS",
        "MENTION",
        "KXEARNINGS",
    ]

    print("\n[1] Searching for earnings-related series...")

    all_series = []

    # Try to get all series and filter
    try:
        # This might need to be adapted based on the actual Kalshi API
        series_list = await client.get_series()

        for series in series_list:
            ticker = series.get("ticker", "")
            title = series.get("title", "")

            # Check if it's earnings-related
            if any(pattern.lower() in ticker.lower() or pattern.lower() in title.lower()
                   for pattern in earnings_series_patterns):
                all_series.append(series)
                print(f"  Found: {ticker} - {title}")

    except Exception as e:
        print(f"  Error getting series: {e}")
        print("  Trying direct series ticker lookup...")

        # Try specific series tickers
        known_series = [
            "KXEARNINGSMENTIONMETA",
            "KXEARNINGSMENTIONTSLA",
            "KXEARNINGSMENTIONNVDA",
            "KXEARNINGSMENTIONAMZN",
            "KXEARNINGSMENTIONAAPL",
            "KXEARNINGSMENTIONMSFT",
            "KXEARNINGSMENTIONgoogl",
        ]

        for series_ticker in known_series:
            try:
                markets = await client.get_markets(series_ticker=series_ticker)
                if markets:
                    print(f"  ✓ Found series: {series_ticker}")
                    all_series.append({"ticker": series_ticker, "markets": markets})
            except Exception as e:
                print(f"  ✗ {series_ticker}: {e}")

    print(f"\n[2] Found {len(all_series)} earnings-related series")

    # Analyze each series
    print("\n[3] Analyzing contracts...")

    contracts_summary = []

    for series in all_series:
        series_ticker = series.get("ticker", "")

        try:
            markets = series.get("markets")
            if not markets:
                markets = await client.get_markets(series_ticker=series_ticker)

            for market in markets[:5]:  # Show first 5
                ticker = market.get("ticker", "")
                title = market.get("title", "")
                status = market.get("status", "")

                print(f"\n  Contract: {ticker}")
                print(f"    Title: {title}")
                print(f"    Status: {status}")

                # Try to extract the tracked word and threshold
                # Expected format: "Will [COMPANY CEO] say '[WORD]' at least [N] times?"
                word_match = re.search(r"say ['\"](.+?)['\"]", title.lower())
                threshold_match = re.search(r"at least (\d+)", title.lower())

                word = word_match.group(1) if word_match else None
                threshold = int(threshold_match.group(1)) if threshold_match else 1

                print(f"    Word: {word}")
                print(f"    Threshold: {threshold}")

                contracts_summary.append({
                    "series": series_ticker,
                    "ticker": ticker,
                    "title": title,
                    "word": word,
                    "threshold": threshold,
                    "status": status,
                })

        except Exception as e:
            print(f"  Error analyzing {series_ticker}: {e}")

    # Save summary
    output_file = Path("data/kalshi_earnings_contracts_summary.json")
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, "w") as f:
        json.dump(contracts_summary, f, indent=2)

    print(f"\n[4] Summary saved to {output_file}")
    print(f"\nTotal contracts found: {len(contracts_summary)}")

    # Print structure info
    if contracts_summary:
        print("\n" + "=" * 60)
        print("EARNINGS MENTION CONTRACT STRUCTURE")
        print("=" * 60)

        # Group by company
        companies = {}
        for contract in contracts_summary:
            # Extract company from series ticker
            series = contract["series"]
            company = series.replace("KXEARNINGSMENTION", "")

            if company not in companies:
                companies[company] = []
            companies[company].append(contract)

        for company, contracts in companies.items():
            print(f"\n{company}:")
            words = set(c["word"] for c in contracts if c["word"])
            print(f"  Words tracked: {', '.join(words)}")
            print(f"  Total contracts: {len(contracts)}")


if __name__ == "__main__":
    import re
    asyncio.run(explore_earnings_contracts())
