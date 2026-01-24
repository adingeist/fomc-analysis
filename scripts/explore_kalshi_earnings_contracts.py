"""
Explore Kalshi earnings mention contracts.

This script checks what earnings call mention contracts are available on Kalshi.
"""

from pathlib import Path
import json
import re

from fomc_analysis.kalshi_client_factory import get_kalshi_client


def explore_earnings_contracts():
    """Explore available earnings mention contracts on Kalshi."""

    # Create client (synchronous - it wraps async internally)
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

    # Try specific series tickers directly
    # (Kalshi doesn't have a list-all-series endpoint)
    print("  Trying direct series ticker lookup...")

    # Try specific series tickers
    known_series = [
        "KXEARNINGSMENTIONMETA",
        "KXEARNINGSMENTIONTSLA",
        "KXEARNINGSMENTIONNVDA",
        "KXEARNINGSMENTIONAMZN",
        "KXEARNINGSMENTIONAAPL",
        "KXEARNINGSMENTIONMSFT",
        "KXEARNINGSMENTIONGOOGLUNNAMED",
    ]

    for series_ticker in known_series:
        try:
            # Client methods are synchronous (no await needed)
            markets = client.get_markets(series_ticker=series_ticker)
            if markets:
                print(f"  ✓ Found series: {series_ticker} ({len(markets)} markets)")
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
                # No await - client is synchronous
                markets = client.get_markets(series_ticker=series_ticker)

            for market in markets[:10]:  # Show first 10
                ticker = market.get("ticker", "")
                title = market.get("title", "")
                status = market.get("status", "")

                # Extract word from custom_strike
                custom_strike = market.get("custom_strike", {})
                word = custom_strike.get("Word", None)

                # Get pricing info
                yes_bid = market.get("yes_bid", 0)
                yes_ask = market.get("yes_ask", 0)
                last_price = market.get("last_price", 0)

                print(f"\n  Contract: {ticker}")
                print(f"    Word: {word}")
                print(f"    Status: {status}")
                print(f"    Last Price: ${last_price/100:.2f} (bid: ${yes_bid/100:.2f}, ask: ${yes_ask/100:.2f})")

                # Contracts are binary (mentioned vs not), threshold is always 1
                contracts_summary.append({
                    "series": series_ticker,
                    "ticker": ticker,
                    "title": title,
                    "word": word,
                    "threshold": 1,  # Binary: mentioned at all
                    "status": status,
                    "yes_bid": yes_bid / 100,  # Convert cents to dollars
                    "yes_ask": yes_ask / 100,
                    "last_price": last_price / 100,
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
    explore_earnings_contracts()
