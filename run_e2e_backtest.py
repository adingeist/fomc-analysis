#!/usr/bin/env python3
"""
End-to-end backtest script with data cleaning and visualization.

This script runs the complete workflow:
1. Optionally cleans data directory
2. Fetches FOMC transcripts
3. Parses into speaker segments
4. Analyzes Kalshi contracts
5. Runs backtest v3
6. Generates visualizations

Usage:
    python run_e2e_backtest.py --clean  # Clean data and run full pipeline
    python run_e2e_backtest.py          # Use existing data if available
"""

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def clean_data_directory(data_dir: Path):
    """Remove all data to start fresh."""
    print("=" * 80)
    print("CLEANING DATA DIRECTORY")
    print("=" * 80)

    if data_dir.exists():
        print(f"\nRemoving {data_dir}...")
        shutil.rmtree(data_dir)
        print("✓ Data directory cleaned")
    else:
        print(f"\n{data_dir} does not exist, skipping clean")

    # Also clean results
    results_dir = Path("results/backtest_v3")
    if results_dir.exists():
        print(f"\nRemoving {results_dir}...")
        shutil.rmtree(results_dir)
        print("✓ Results directory cleaned")


def run_command(cmd: List[str], description: str) -> bool:
    """Run a shell command and return success status."""
    print(f"\n{'=' * 80}")
    print(description)
    print(f"{'=' * 80}")

    # Replace fomc-analysis with fomc (correct command name)
    if cmd[0] == "fomc-analysis":
        cmd[0] = "fomc"

    print(f"\nCommand: {' '.join(cmd)}")
    print()

    try:
        result = subprocess.run(
            cmd,
            check=True,
            capture_output=False,
            text=True
        )
        print(f"\n✓ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n✗ {description} failed with exit code {e.returncode}")
        print(f"Error: {e}")
        return False
    except FileNotFoundError:
        print(f"\n✗ Command not found: {cmd[0]}")
        print("Make sure fomc-analysis is installed: pip install -e .")
        return False


def fetch_transcripts(start_year: int, end_year: int) -> bool:
    """Fetch FOMC transcripts."""
    return run_command(
        [
            "fomc-analysis",
            "fetch-transcripts",
            "--start-year", str(start_year),
            "--end-year", str(end_year),
            "--out-dir", "data/raw_pdf",
        ],
        "STEP 1: Fetching FOMC Transcripts"
    )


def parse_transcripts() -> bool:
    """Parse transcripts into speaker segments."""
    return run_command(
        [
            "fomc-analysis",
            "parse",
            "--input-dir", "data/raw_pdf",
            "--mode", "deterministic",
            "--segments-dir", "data/segments",
        ],
        "STEP 2: Parsing Transcripts"
    )


def analyze_kalshi_contracts(series_ticker: str, market_status: str) -> bool:
    """Analyze Kalshi contracts and generate contract words."""
    return run_command(
        [
            "fomc-analysis",
            "analyze-kalshi-contracts",
            "--series-ticker", series_ticker,
            "--segments-dir", "data/segments",
            "--output-dir", "data/kalshi_analysis",
            "--scope", "powell_only",
            "--market-status", market_status,
        ],
        "STEP 3: Analyzing Kalshi Contracts"
    )


def run_backtest(
    edge_threshold: float,
    position_size: float,
    initial_capital: float,
    horizons: str,
    alpha: float,
    beta_prior: float,
    half_life: int,
    prior_strength: float,
    min_history: int,
    min_yes_prob: float,
    max_no_prob: float,
    fee_rate: float,
    transaction_cost: float,
    slippage: float,
    yes_edge_threshold: Optional[float] = None,
    no_edge_threshold: Optional[float] = None,
    yes_position_size: Optional[float] = None,
    no_position_size: Optional[float] = None,
    max_position_size: Optional[float] = None,
    train_window: Optional[int] = None,
    test_start_date: Optional[str] = None,
) -> bool:
    """Run backtest v3."""
    cmd = [
        "fomc-analysis",
        "backtest-v3",
        "--contract-words", "data/kalshi_analysis/contract_words.json",
        "--segments-dir", "data/segments",
        "--model", "beta",
        "--alpha", str(alpha),
        "--beta-prior", str(beta_prior),
        "--half-life", str(half_life),
        "--prior-strength", str(prior_strength),
        "--min-history", str(min_history),
        "--horizons", horizons,
        "--edge-threshold", str(edge_threshold),
        "--position-size-pct", str(position_size),
        "--min-yes-prob", str(min_yes_prob),
        "--max-no-prob", str(max_no_prob),
        "--initial-capital", str(initial_capital),
        "--fee-rate", str(fee_rate),
        "--transaction-cost", str(transaction_cost),
        "--slippage", str(slippage),
        "--output", "results/backtest_v3",
    ]

    if yes_edge_threshold is not None:
        cmd.extend(["--yes-edge-threshold", str(yes_edge_threshold)])
    if no_edge_threshold is not None:
        cmd.extend(["--no-edge-threshold", str(no_edge_threshold)])
    if yes_position_size is not None:
        cmd.extend(["--yes-position-size-pct", str(yes_position_size)])
    if no_position_size is not None:
        cmd.extend(["--no-position-size-pct", str(no_position_size)])
    if max_position_size is not None:
        cmd.extend(["--max-position-size", str(max_position_size)])
    if train_window is not None:
        cmd.extend(["--train-window-size", str(train_window)])
    if test_start_date:
        cmd.extend(["--test-start-date", test_start_date])

    return run_command(cmd, "STEP 4: Running Backtest v3")


def run_upcoming_predictions(
    alpha: float,
    beta_prior: float,
    half_life: int,
    prior_strength: float,
    min_history: int,
    output_dir: Path,
) -> bool:
    """Generate live predictions for unresolved contracts."""
    return run_command(
        [
            "fomc-analysis",
            "predict-upcoming",
            "--contract-words", "data/kalshi_analysis/contract_words.json",
            "--alpha", str(alpha),
            "--beta-prior", str(beta_prior),
            "--half-life", str(half_life),
            "--prior-strength", str(prior_strength),
            "--min-history", str(min_history),
            "--output", str(output_dir),
        ],
        "STEP 5: Generating Upcoming Predictions",
    )


def load_backtest_results(results_dir: Path) -> Dict:
    """Load backtest results."""
    results_file = results_dir / "backtest_results.json"
    if not results_file.exists():
        raise FileNotFoundError(f"Results file not found: {results_file}")

    return json.loads(results_file.read_text())


def get_traded_contracts(results: Dict) -> List[str]:
    """Extract list of contracts that were actually traded."""
    trades = results.get("trades", [])
    if not trades:
        return []

    contracts = set(trade["contract"] for trade in trades)
    return sorted(contracts)


def load_segment_data(segments_dir: Path) -> pd.DataFrame:
    """Load all segment files and extract Powell mentions."""
    from fomc_analysis.parsing.speaker_segmenter import load_segments_jsonl

    records = []

    for segment_file in sorted(segments_dir.glob("*.jsonl")):
        date_str = segment_file.stem

        try:
            segments = load_segments_jsonl(segment_file)
        except Exception as e:
            print(f"Warning: Could not load {segment_file}: {e}")
            continue

        # Get Powell's text
        powell_segments = [s for s in segments if s.role == "powell"]
        powell_text = " ".join(seg.text for seg in powell_segments)

        records.append({
            "date": pd.to_datetime(date_str, format="%Y%m%d"),
            "text": powell_text,
        })

    return pd.DataFrame(records).sort_values("date")


def count_contract_mentions(
    segments_df: pd.DataFrame,
    contract_words: Dict,
) -> pd.DataFrame:
    """Count mentions of each contract in each transcript."""
    from fomc_analysis.featurizer import count_phrase_mentions

    mention_data = []

    for _, row in segments_df.iterrows():
        text = row["text"]
        date = row["date"]

        for contract_name, contract_def in contract_words.items():
            # Get synonyms/phrases for this contract
            synonyms = contract_def.get("synonyms", [])
            if not synonyms:
                continue

            # Count mentions
            count = count_phrase_mentions(
                text=text,
                phrases=synonyms,
                case_sensitive=False,
                word_boundaries=True,
            )

            mention_data.append({
                "date": date,
                "contract": contract_name,
                "count": count,
            })

    mentions_df = pd.DataFrame(mention_data)
    return mentions_df.pivot_table(
        index="date",
        columns="contract",
        values="count",
        aggfunc="sum",
        fill_value=0
    )


def generate_frequency_chart(
    mentions_df: pd.DataFrame,
    traded_contracts: List[str],
    output_path: Path,
):
    """Generate line chart of word frequencies for traded contracts."""
    print("\n" + "=" * 80)
    print("GENERATING VISUALIZATION")
    print("=" * 80)

    # Filter to only traded contracts
    available_contracts = [c for c in traded_contracts if c in mentions_df.columns]

    if not available_contracts:
        print("\n✗ No traded contracts found in mentions data")
        return

    plot_data = mentions_df[available_contracts]

    # Create figure
    fig, ax = plt.subplots(figsize=(14, 8))

    # Plot each contract
    for contract in available_contracts:
        ax.plot(
            plot_data.index,
            plot_data[contract],
            marker='o',
            linewidth=2,
            markersize=6,
            label=contract,
            alpha=0.7,
        )

    # Styling
    ax.set_xlabel("FOMC Meeting Date", fontsize=12, fontweight='bold')
    ax.set_ylabel("Mention Count (Powell Only)", fontsize=12, fontweight='bold')
    ax.set_title(
        "Word Frequencies for Traded Contracts Across FOMC Meetings",
        fontsize=14,
        fontweight='bold',
        pad=20
    )
    ax.legend(
        loc='upper left',
        bbox_to_anchor=(1.02, 1),
        fontsize=10,
        frameon=True,
        shadow=True
    )
    ax.grid(True, alpha=0.3, linestyle='--')

    # Format x-axis dates
    import matplotlib.dates as mdates
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    plt.xticks(rotation=45, ha='right')

    # Tight layout
    plt.tight_layout()

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Chart saved to: {output_path}")

    # Also save as PNG
    png_path = output_path.with_suffix('.png')
    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    print(f"✓ Chart saved to: {png_path}")

    plt.close()


def generate_summary_stats(mentions_df: pd.DataFrame, traded_contracts: List[str]):
    """Generate and print summary statistics."""
    print("\n" + "=" * 80)
    print("WORD FREQUENCY STATISTICS")
    print("=" * 80)

    available_contracts = [c for c in traded_contracts if c in mentions_df.columns]

    if not available_contracts:
        print("\n✗ No traded contracts found in mentions data")
        return

    stats_data = []
    for contract in available_contracts:
        data = mentions_df[contract]
        stats_data.append({
            "Contract": contract,
            "Mean": f"{data.mean():.2f}",
            "Median": f"{data.median():.2f}",
            "Std Dev": f"{data.std():.2f}",
            "Min": int(data.min()),
            "Max": int(data.max()),
            "Total Mentions": int(data.sum()),
        })

    stats_df = pd.DataFrame(stats_data)
    print("\n" + stats_df.to_string(index=False))


def print_final_summary(results: Dict):
    """Print final backtest summary."""
    print("\n" + "=" * 80)
    print("FINAL BACKTEST SUMMARY")
    print("=" * 80)

    overall = results["overall_metrics"]
    print(f"""
Overall Performance:
  Total Trades:     {overall['total_trades']}
  Win Rate:         {overall['win_rate'] * 100:.1f}%
  Total P&L:        ${overall['total_pnl']:,.2f}
  ROI:              {overall['roi'] * 100:.1f}%
  Sharpe Ratio:     {overall['sharpe']:.2f}
  Final Capital:    ${overall['final_capital']:,.2f}
""")

    print("\nPerformance by Time Horizon:")
    for horizon, metrics in sorted(results["horizon_metrics"].items(), key=lambda x: int(x[0])):
        print(f"""
  {horizon} days before meeting:
    Predictions:      {metrics['total_predictions']}
    Accuracy:         {metrics['accuracy'] * 100:.1f}%
    Trades:           {metrics['total_trades']}
    Win Rate:         {metrics['win_rate'] * 100:.1f}%
    Total P&L:        ${metrics['total_pnl']:,.2f}
    ROI:              {metrics['roi'] * 100:.1f}%
    Brier Score:      {metrics['brier_score']:.3f}
""")


def main():
    parser = argparse.ArgumentParser(
        description="End-to-end backtest with optional data cleaning and visualization"
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Clean data directory before running"
    )
    parser.add_argument(
        "--start-year",
        type=int,
        default=2020,
        help="Start year for transcript fetch (default: 2020)"
    )
    parser.add_argument(
        "--end-year",
        type=int,
        default=2025,
        help="End year for transcript fetch (default: 2025)"
    )
    parser.add_argument(
        "--series-ticker",
        type=str,
        default="KXFEDMENTION",
        help="Kalshi series ticker (default: KXFEDMENTION)"
    )
    parser.add_argument(
        "--market-status",
        type=str,
        default="all",
        help="Kalshi market status filter for analysis (default: all)"
    )
    parser.add_argument(
        "--edge-threshold",
        type=float,
        default=0.12,
        help="Edge threshold for trading (default: 0.12)"
    )
    parser.add_argument(
        "--yes-edge-threshold",
        type=float,
        default=0.20,
        help="Edge requirement for YES trades (default: 0.20)"
    )
    parser.add_argument(
        "--no-edge-threshold",
        type=float,
        default=0.08,
        help="Edge requirement for NO trades (default: 0.08)"
    )
    parser.add_argument(
        "--position-size",
        type=float,
        default=0.05,
        help="Position size as fraction of capital (default: 0.05)"
    )
    parser.add_argument(
        "--yes-position-size",
        type=float,
        default=0.04,
        help="Position size fraction override for YES trades (default: 0.04)"
    )
    parser.add_argument(
        "--no-position-size",
        type=float,
        default=0.03,
        help="Position size fraction override for NO trades (default: 0.03)"
    )
    parser.add_argument(
        "--max-position-size",
        type=float,
        default=1500.0,
        help="Hard cap on dollars per trade (default: 1500)"
    )
    parser.add_argument(
        "--initial-capital",
        type=float,
        default=10000.0,
        help="Initial capital (default: 10000)"
    )
    parser.add_argument(
        "--horizons",
        type=str,
        default="7,14,30",
        help="Prediction horizons in days (default: 7,14,30)"
    )
    parser.add_argument(
        "--model-alpha",
        type=float,
        default=1.0,
        help="Alpha prior for the Beta-Binomial model (default: 1.0)"
    )
    parser.add_argument(
        "--beta-prior",
        type=float,
        default=1.0,
        help="Beta prior parameter for the Beta-Binomial model (default: 1.0)"
    )
    parser.add_argument(
        "--half-life",
        type=int,
        default=4,
        help="Half-life for exponential decay (default: 4 meetings)"
    )
    parser.add_argument(
        "--model-prior-strength",
        type=float,
        default=4.0,
        help="Strength of contract-level pseudo-counts (default: 4.0)"
    )
    parser.add_argument(
        "--model-min-history",
        type=int,
        default=5,
        help="Minimum observed meetings before using full weight (default: 5)"
    )
    parser.add_argument(
        "--min-yes-prob",
        type=float,
        default=0.65,
        help="Only place YES trades when model probability exceeds this value (default: 0.65)"
    )
    parser.add_argument(
        "--max-no-prob",
        type=float,
        default=0.35,
        help="Only place NO trades when model probability is below this value (default: 0.35)"
    )
    parser.add_argument(
        "--fee-rate",
        type=float,
        default=0.07,
        help="Kalshi fee rate on profits (default: 0.07)"
    )
    parser.add_argument(
        "--transaction-cost",
        type=float,
        default=0.01,
        help="Additional transaction cost per trade as fraction of position size (default: 0.01)"
    )
    parser.add_argument(
        "--slippage",
        type=float,
        default=0.01,
        help="Slippage applied to entry price (price units, default: 0.01)"
    )
    parser.add_argument(
        "--train-window",
        type=int,
        default=12,
        help="Rolling number of meetings to use for training (default: 12)"
    )
    parser.add_argument(
        "--test-start-date",
        type=str,
        default="2022-01-01",
        help="Only score meetings on/after this date (YYYY-MM-DD, default: 2022-01-01)"
    )
    parser.add_argument(
        "--skip-fetch",
        action="store_true",
        help="Skip transcript fetch (use existing data)"
    )
    parser.add_argument(
        "--skip-parse",
        action="store_true",
        help="Skip parsing (use existing segments)"
    )
    parser.add_argument(
        "--skip-analyze",
        action="store_true",
        help="Skip Kalshi analysis (use existing contract_words.json)"
    )
    parser.add_argument(
        "--skip-predict",
        action="store_true",
        help="Skip generating upcoming predictions"
    )
    parser.add_argument(
        "--prediction-output",
        type=Path,
        default=Path("results/upcoming_predictions"),
        help="Directory to store upcoming predictions (default: results/upcoming_predictions)"
    )

    args = parser.parse_args()

    print("=" * 80)
    print("FOMC ANALYSIS - END-TO-END BACKTEST WITH VISUALIZATION")
    print("=" * 80)
    print(f"""
Configuration:
  Years:            {args.start_year} - {args.end_year}
  Series:           {args.series_ticker}
  Market Status:    {args.market_status}
  Edge Threshold:   {args.edge_threshold * 100:.0f}%
  Position Size:    {args.position_size * 100:.0f}%
  Initial Capital:  ${args.initial_capital:,.2f}
  Horizons:         {args.horizons} days
  Beta Alpha:       {args.model_alpha}
  Beta Beta:        {args.beta_prior}
  Half-life:        {args.half_life} meetings
  Prior Strength:   {args.model_prior_strength}
  Min History:      {args.model_min_history}
  YES Threshold:    {args.min_yes_prob}
  NO Threshold:     {args.max_no_prob}
  Clean Data:       {args.clean}
""")

    # Step 0: Clean data if requested
    if args.clean:
        clean_data_directory(Path("data"))

    # Step 1: Fetch transcripts
    if not args.skip_fetch:
        if not fetch_transcripts(args.start_year, args.end_year):
            print("\n✗ Pipeline failed at transcript fetch")
            return 1
    else:
        print("\n→ Skipping transcript fetch (--skip-fetch)")

    # Step 2: Parse transcripts
    if not args.skip_parse:
        if not parse_transcripts():
            print("\n✗ Pipeline failed at parsing")
            return 1
    else:
        print("\n→ Skipping parse (--skip-parse)")

    # Step 3: Analyze Kalshi contracts
    if not args.skip_analyze:
        if not analyze_kalshi_contracts(args.series_ticker, args.market_status):
            print("\n✗ Pipeline failed at Kalshi analysis")
            return 1
    else:
        print("\n→ Skipping Kalshi analysis (--skip-analyze)")

    # Step 4: Run backtest
    if not run_backtest(
        args.edge_threshold,
        args.position_size,
        args.initial_capital,
        args.horizons,
        args.model_alpha,
        args.beta_prior,
        args.half_life,
        args.model_prior_strength,
        args.model_min_history,
        args.min_yes_prob,
        args.max_no_prob,
        args.fee_rate,
        args.transaction_cost,
        args.slippage,
        args.yes_edge_threshold,
        args.no_edge_threshold,
        args.yes_position_size,
        args.no_position_size,
        args.max_position_size,
        args.train_window,
        args.test_start_date,
    ):
        print("\n✗ Pipeline failed at backtest")
        return 1

    # Step 5: Generate upcoming predictions
    if args.skip_predict:
        print("\n→ Skipping upcoming predictions (--skip-predict)")
    else:
        prediction_output = args.prediction_output
        if not run_upcoming_predictions(
            args.model_alpha,
            args.beta_prior,
            args.half_life,
            args.model_prior_strength,
            args.model_min_history,
            prediction_output,
        ):
            print("\n✗ Pipeline failed at upcoming predictions")
            return 1

    # Step 6: Generate visualizations
    try:
        results_dir = Path("results/backtest_v3")
        results = load_backtest_results(results_dir)

        # Get traded contracts
        traded_contracts = get_traded_contracts(results)
        print(f"\n✓ Found {len(traded_contracts)} traded contracts")

        if traded_contracts:
            # Load contract mapping
            contract_words_file = Path("data/kalshi_analysis/generated_contract_mapping.yaml")
            if not contract_words_file.exists():
                contract_words_file = Path("configs/generated_contract_mapping.yaml")

            if contract_words_file.exists():
                import yaml
                contract_words = yaml.safe_load(contract_words_file.read_text())

                # Load segment data
                print("\nLoading segment data for frequency analysis...")
                segments_df = load_segment_data(Path("data/segments"))

                # Count mentions
                print("Counting contract mentions...")
                mentions_df = count_contract_mentions(segments_df, contract_words)

                # Generate chart
                generate_frequency_chart(
                    mentions_df,
                    traded_contracts,
                    results_dir / "word_frequencies.pdf"
                )

                # Print statistics
                generate_summary_stats(mentions_df, traded_contracts)
            else:
                print(f"\n✗ Contract mapping not found at {contract_words_file}")
        else:
            print("\n→ No trades executed, skipping frequency chart")

        # Print final summary
        print_final_summary(results)

    except Exception as e:
        print(f"\n✗ Visualization generation failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

    # Success!
    print("\n" + "=" * 80)
    print("✓ END-TO-END PIPELINE COMPLETED SUCCESSFULLY!")
    print("=" * 80)
    print(f"""
Results saved to: results/backtest_v3/
  - backtest_results.json
  - predictions.csv
  - trades.csv
  - horizon_metrics.csv
  - word_frequencies.pdf
  - word_frequencies.png
""")

    return 0


if __name__ == "__main__":
    sys.exit(main())
