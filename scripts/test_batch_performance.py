"""
Test script to compare sequential vs parallel API fetching performance.

This demonstrates the performance improvement from using asyncio.gather and batching.
"""

import asyncio
import time
from fomc_analysis.kalshi_client_factory import (
    get_kalshi_client,
    batch_get_markets,
    KalshiSdkAdapter,
)


async def test_sequential_fetching(client, series_tickers):
    """Fetch markets sequentially (old approach)."""
    start_time = time.time()

    all_markets = []
    for ticker in series_tickers:
        try:
            markets = client.get_markets(series_ticker=ticker)
            all_markets.append((ticker, markets))
        except Exception as e:
            print(f"Error fetching {ticker}: {e}")
            all_markets.append((ticker, []))

    elapsed = time.time() - start_time
    total_markets = sum(len(markets) for _, markets in all_markets)

    return elapsed, total_markets


async def test_parallel_fetching(client, series_tickers):
    """Fetch markets in parallel using asyncio.gather (new approach)."""
    if not isinstance(client, KalshiSdkAdapter):
        print("Client is not KalshiSdkAdapter, cannot test parallel fetching")
        return None, 0

    start_time = time.time()

    markets_by_series = await batch_get_markets(
        client=client,
        series_tickers=series_tickers,
    )

    elapsed = time.time() - start_time
    total_markets = sum(len(markets) for markets in markets_by_series.values())

    return elapsed, total_markets


async def main():
    """Run performance comparison test."""
    print("=" * 70)
    print("Kalshi API Batch Fetching Performance Test")
    print("=" * 70)

    # Get client
    client = get_kalshi_client()

    # Test with multiple series
    test_series = [
        "KXEARNINGSMENTIONMETA",
        "KXEARNINGSMENTIONTSLA",
        "KXEARNINGSMENTIONNVDA",
        "KXEARNINGSMENTIONAMZN",
        "KXEARNINGSMENTIONAAPL",
        "KXEARNINGSMENTIONMSFT",
    ]

    print(f"\nFetching contracts for {len(test_series)} series...")
    print(f"Series: {', '.join(test_series)}\n")

    # Test sequential fetching
    print("[1] Sequential Fetching (old approach)")
    seq_time, seq_markets = await test_sequential_fetching(client, test_series)
    print(f"    Time: {seq_time:.2f}s")
    print(f"    Markets fetched: {seq_markets}")

    # Test parallel fetching
    print("\n[2] Parallel Fetching with asyncio.gather (new approach)")
    par_time, par_markets = await test_parallel_fetching(client, test_series)

    if par_time is not None:
        print(f"    Time: {par_time:.2f}s")
        print(f"    Markets fetched: {par_markets}")

        # Calculate speedup
        speedup = seq_time / par_time if par_time > 0 else 0
        time_saved = seq_time - par_time

        print("\n" + "=" * 70)
        print("RESULTS")
        print("=" * 70)
        print(f"Sequential time:  {seq_time:.2f}s")
        print(f"Parallel time:    {par_time:.2f}s")
        print(f"Speedup:          {speedup:.2f}x")
        print(f"Time saved:       {time_saved:.2f}s ({time_saved/seq_time*100:.1f}%)")
        print("=" * 70)
    else:
        print("    Parallel fetching not available (client type not supported)")

    # Close client
    if hasattr(client, 'close'):
        client.close()


if __name__ == "__main__":
    asyncio.run(main())
