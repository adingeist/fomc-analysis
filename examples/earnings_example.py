"""
Example: Complete earnings call analysis workflow.

This script demonstrates how to:
1. Fetch earnings transcripts for a ticker
2. Fetch price outcomes
3. Parse and segment transcripts
4. Extract features
5. Run backtest
"""

import os
from pathlib import Path

import pandas as pd

from earnings_analysis.fetchers import (
    fetch_earnings_transcripts,
    fetch_price_outcomes,
    fetch_earnings_data,
)
from earnings_analysis.parsing import segment_earnings_transcript, parse_transcript
from earnings_analysis.features import featurize_earnings_calls
from earnings_analysis.models import BetaBinomialEarningsModel
from earnings_analysis.backtester import EarningsBacktester, save_backtest_result


def main():
    """Run complete earnings analysis example."""

    # Configuration
    TICKER = "COIN"  # Coinbase
    NUM_QUARTERS = 8
    OUTPUT_DIR = Path("data/earnings_example")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"=" * 60)
    print(f"Earnings Analysis Example: {TICKER}")
    print(f"=" * 60)

    # Step 1: Fetch earnings data and dates
    print("\n[1/6] Fetching earnings data...")

    earnings_data = fetch_earnings_data(TICKER)

    if earnings_data.empty:
        print(f"No earnings data found for {TICKER}")
        return

    print(f"Found {len(earnings_data)} earnings dates")

    # Get earnings dates
    earnings_dates = pd.to_datetime(earnings_data["date"]).tolist()

    # Step 2: Fetch price outcomes
    print("\n[2/6] Fetching price outcomes...")

    outcomes = fetch_price_outcomes(
        ticker=TICKER,
        earnings_dates=earnings_dates[:NUM_QUARTERS],
        horizons=[1, 5, 10],
    )

    if outcomes.empty:
        print("No price data found")
        return

    print(f"Fetched outcomes for {len(outcomes)} earnings calls")

    # Save outcomes
    outcomes_file = OUTPUT_DIR / f"{TICKER}_outcomes.csv"
    outcomes.to_csv(outcomes_file, index=False)
    print(f"Saved outcomes to {outcomes_file}")

    # Step 3: Fetch transcripts (placeholder - requires actual transcript source)
    print("\n[3/6] Fetching transcripts...")
    print("Note: SEC EDGAR transcripts require actual 8-K filings.")
    print("For this example, we'll simulate with dummy data.")

    # In a real scenario, you would:
    # metadata_list = fetch_earnings_transcripts(
    #     ticker=TICKER,
    #     output_dir=OUTPUT_DIR / "transcripts",
    #     num_quarters=NUM_QUARTERS,
    #     source="sec",
    # )

    # Step 4: Parse transcripts (skipped in this example)
    print("\n[4/6] Parsing transcripts...")
    print("Skipped - requires actual transcript files")

    # Step 5: Featurize (using dummy features for demonstration)
    print("\n[5/6] Creating feature matrix...")

    # For this example, create dummy features
    features = pd.DataFrame({
        "call_date": outcomes["earnings_date"],
        "ticker": TICKER,
        "sentiment_positive_count": pd.Series([10, 15, 8, 12, 14, 9, 11, 13][:len(outcomes)]),
        "sentiment_negative_count": pd.Series([5, 3, 7, 4, 2, 6, 5, 3][:len(outcomes)]),
        "guidance_count": pd.Series([8, 10, 6, 9, 11, 7, 8, 10][:len(outcomes)]),
        "ceo_word_count": pd.Series([500, 550, 480, 520, 560, 490, 510, 540][:len(outcomes)]),
        "cfo_word_count": pd.Series([400, 420, 380, 410, 430, 390, 400, 420][:len(outcomes)]),
    })

    features["call_date"] = pd.to_datetime(features["call_date"])
    features = features.set_index("call_date")

    print(f"Feature matrix: {features.shape}")

    # Step 6: Run backtest
    print("\n[6/6] Running backtest...")

    backtester = EarningsBacktester(
        features=features,
        outcomes=outcomes.rename(columns={"earnings_date": "call_date"}).set_index("call_date"),
        model_class=BetaBinomialEarningsModel,
        model_params={"alpha_prior": 1.0, "beta_prior": 1.0, "half_life": 4},
        edge_threshold=0.1,
        position_size_pct=0.05,
        min_train_window=3,
    )

    result = backtester.run(initial_capital=10000)

    # Save results
    results_dir = OUTPUT_DIR / "backtest_results"
    save_backtest_result(result, results_dir)

    print(f"\nBacktest complete!")
    print(f"Results saved to {results_dir}")

    # Print metrics
    print("\n" + "=" * 60)
    print("BACKTEST METRICS")
    print("=" * 60)
    for key, value in result.metrics.items():
        if isinstance(value, float):
            print(f"{key:20s}: {value:>10.4f}")
        else:
            print(f"{key:20s}: {value:>10}")

    print("\n" + "=" * 60)
    print(f"Total Predictions: {len(result.predictions)}")
    print(f"Total Trades: {len(result.trades)}")

    if result.predictions:
        correct_predictions = sum(1 for p in result.predictions if p.correct)
        accuracy = correct_predictions / len(result.predictions)
        print(f"Prediction Accuracy: {accuracy:.2%}")

    print("=" * 60)


if __name__ == "__main__":
    main()
