"""
Explore Kalshi earnings mention contracts.

This script checks what earnings call mention contracts are available on Kalshi.
Uses async batching for performance.
"""

from pathlib import Path
import json
import re
import asyncio

from fomc_analysis.kalshi_client_factory import (
    get_kalshi_client,
    batch_get_markets,
    KalshiSdkAdapter,
)


async def explore_earnings_contracts_async():
    """Explore available earnings mention contracts on Kalshi using async batching."""

    # Create client
    client = get_kalshi_client()

    if not isinstance(client, KalshiSdkAdapter):
        print("Warning: Client is not KalshiSdkAdapter, falling back to sequential fetching")
        # Fallback to sync version
        explore_earnings_contracts_sync()
        return

    print("=" * 60)
    print("Exploring Kalshi Earnings Mention Contracts (Async Batching)")
    print("=" * 60)

    print("\n[1] Fetching earnings contracts in parallel...")

    # Try specific series tickers
    known_series = [
        "KXEARNINGSMENTIONMETA",
        "KXEARNINGSMENTIONTSLA",
        "KXEARNINGSMENTIONNVDA",
        "KXEARNINGSMENTIONAMZN",
        "KXEARNINGSMENTIONAAPL",
        "KXEARNINGSMENTIONMSFT",
        "KXEARNINGSMENTIONGOOGLUNNAMED",
        "KXEARNINGSMENTIONAMZN",
        "KXEARNINGSMENTIONNETFLIX",
        "KXEARNINGSMENTIONNFLX",
    ]

    # Fetch all series in parallel using asyncio.gather
    markets_by_series = await batch_get_markets(
        client=client,
        series_tickers=known_series,
        status=None,  # All statuses
        limit=200,
    )

    # Build all_series list
    all_series = []
    for series_ticker, markets in markets_by_series.items():
        if markets:
            print(f"  ✓ Found series: {series_ticker} ({len(markets)} markets)")
            all_series.append({"ticker": series_ticker, "markets": markets})
        else:
            print(f"  ✗ {series_ticker}: No markets found")

    print(f"\n[2] Found {len(all_series)} earnings-related series with markets")

    # Analyze each series
    print("\n[3] Analyzing contracts...")

    contracts_summary = []

    for series in all_series:
        series_ticker = series.get("ticker", "")

        try:
            markets = series.get("markets", [])

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


def explore_earnings_contracts_sync():
    """Synchronous fallback version for non-SDK clients."""
    client = get_kalshi_client()

    print("=" * 60)
    print("Exploring Kalshi Earnings Mention Contracts (Sequential)")
    print("=" * 60)

    print("\n[1] Searching for earnings-related series...")

    all_series = []

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
            markets = client.get_markets(series_ticker=series_ticker)
            if markets:
                print(f"  ✓ Found series: {series_ticker} ({len(markets)} markets)")
                all_series.append({"ticker": series_ticker, "markets": markets})
        except Exception as e:
            print(f"  ✗ {series_ticker}: {e}")

    print(f"\n[2] Found {len(all_series)} earnings-related series")

    # Analyze each series (same as async version)
    print("\n[3] Analyzing contracts...")

    contracts_summary = []

    for series in all_series:
        series_ticker = series.get("ticker", "")

        try:
            markets = series.get("markets", [])

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

                contracts_summary.append({
                    "series": series_ticker,
                    "ticker": ticker,
                    "title": title,
                    "word": word,
                    "threshold": 1,
                    "status": status,
                    "yes_bid": yes_bid / 100,
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


def explore_earnings_contracts():
    """Main entry point - uses async if possible, sync otherwise."""
    try:
        asyncio.run(explore_earnings_contracts_async())
    except Exception as e:
        print(f"Async version failed ({e}), falling back to sync")
        explore_earnings_contracts_sync()


if __name__ == "__main__":
    explore_earnings_contracts()
