#!/usr/bin/env python3
"""
Run backtest with expanded training data from historical press releases.

This is the FOMC-style approach: use press releases from quarters that
predate Kalshi contracts to build a larger training set. Instead of 2
training events, we get 5-8 — giving the Beta-Binomial model a much
tighter estimate of word-mention probabilities.

Press releases are a conservative source (they undercount mentions
compared to the full earnings call), so confirmed-YES from a press
release is highly reliable, while absence is ambiguous.

Prerequisites:
    python scripts/build_ground_truth.py              # real outcomes
    python scripts/fetch_real_transcripts.py --mode dates --ticker META  # etc.

Usage:
    python scripts/run_expanded_backtest.py
    python scripts/run_expanded_backtest.py --company META
    python scripts/run_expanded_backtest.py --absent-means-no  # strict mode
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from earnings_analysis.ground_truth import (
    fetch_ground_truth,
    save_ground_truth,
    load_ground_truth,
    build_backtest_dataframes,
)
from earnings_analysis.synthetic_history import (
    build_expanded_training_data,
    print_expanded_data_summary,
)
from earnings_analysis.models.beta_binomial import BetaBinomialEarningsModel
from earnings_analysis.kalshi.backtester import (
    EarningsKalshiBacktester,
    BacktestResult,
    save_earnings_backtest_result,
)


def run_expanded_backtest(
    ticker: str,
    edge_threshold: float = 0.12,
    half_life: float = 8.0,
    min_train_window: int = 3,
    initial_capital: float = 10000.0,
    absent_means_no: bool = False,
    ground_truth_dir: Path = Path("data/ground_truth"),
    pr_dir: Path = Path("data/earnings/press_releases"),
    transcripts_dir: Path = Path("data/earnings/transcripts"),
) -> tuple[BacktestResult | None, dict]:
    """Run backtest for one company using expanded training data."""

    # Build expanded dataset
    features, outcomes, diagnostics = build_expanded_training_data(
        ticker=ticker,
        ground_truth_dir=ground_truth_dir,
        pr_dir=pr_dir,
        transcripts_dir=transcripts_dir,
        absent_in_pr_means_no=absent_means_no,
    )

    if outcomes.empty:
        diagnostics["error"] = "No data found"
        return None, diagnostics

    print_expanded_data_summary(ticker, outcomes, diagnostics)

    # For Beta-Binomial training with NaN values, we need to handle them.
    # NaN means "unknown" from press releases. Two approaches:
    #   1. Drop NaN rows/columns (lose data)
    #   2. Only train on non-NaN observations per word
    # The backtester handles this per-word, so NaN values in the outcomes
    # DataFrame just reduce the effective training set for that word.

    if len(outcomes) < min_train_window + 1:
        diagnostics["error"] = (
            f"Need at least {min_train_window + 1} dates, have {len(outcomes)}"
        )
        return None, diagnostics

    model_params = {
        "alpha_prior": 1.0,
        "beta_prior": 1.0,
        "half_life": half_life,
    }

    backtester = EarningsKalshiBacktester(
        features=features,
        outcomes=outcomes,
        model_class=BetaBinomialEarningsModel,
        model_params=model_params,
        edge_threshold=edge_threshold,
        min_train_window=min_train_window,
        require_variation=False,
    )

    result = backtester.run(
        ticker=ticker,
        initial_capital=initial_capital,
        market_prices=None,  # no pre-settlement prices available
    )

    return result, diagnostics


def run_sparse_backtest(
    ticker: str,
    edge_threshold: float = 0.12,
    half_life: float = 8.0,
    min_train_window: int = 1,
    initial_capital: float = 10000.0,
    ground_truth_dir: Path = Path("data/ground_truth"),
) -> tuple[BacktestResult | None, dict]:
    """Run backtest using only real Kalshi outcomes (for comparison)."""

    gt_path = ground_truth_dir / "ground_truth.json"
    if not gt_path.exists():
        return None, {"error": "No ground truth data"}

    dataset = load_ground_truth(ground_truth_dir)
    features, outcomes, _ = build_backtest_dataframes(dataset, ticker)

    if outcomes.empty or len(outcomes) < min_train_window + 1:
        return None, {"error": f"Not enough real data ({len(outcomes)} dates)"}

    model_params = {
        "alpha_prior": 1.0,
        "beta_prior": 1.0,
        "half_life": half_life,
    }

    backtester = EarningsKalshiBacktester(
        features=features,
        outcomes=outcomes,
        model_class=BetaBinomialEarningsModel,
        model_params=model_params,
        edge_threshold=edge_threshold,
        min_train_window=min_train_window,
        require_variation=False,
    )

    result = backtester.run(
        ticker=ticker,
        initial_capital=initial_capital,
        market_prices=None,
    )

    return result, {"ticker": ticker, "dates": len(outcomes)}


def main():
    parser = argparse.ArgumentParser(
        description="Run backtest with expanded training data from press releases"
    )
    parser.add_argument("--company", type=str, help="Single company ticker")
    parser.add_argument("--edge-threshold", type=float, default=0.10)
    parser.add_argument("--half-life", type=float, default=8.0)
    parser.add_argument("--min-train-window", type=int, default=3)
    parser.add_argument("--initial-capital", type=float, default=10000.0)
    parser.add_argument(
        "--absent-means-no",
        action="store_true",
        help="Treat words absent from press releases as NO (strict mode). "
             "Default: treat as unknown (NaN).",
    )
    parser.add_argument("--ground-truth-dir", type=Path, default=Path("data/ground_truth"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/expanded_backtest"))
    args = parser.parse_args()

    print("=" * 70)
    print("EXPANDED BACKTEST: SYNTHETIC HISTORY + REAL OUTCOMES")
    print("=" * 70)
    print(f"\nApproach: FOMC-style — scan historical press releases for tracked")
    print(f"words to build training data predating Kalshi contracts.")
    print(f"\nParameters:")
    print(f"  Edge threshold:    {args.edge_threshold}")
    print(f"  Half-life:         {args.half_life}")
    print(f"  Min train window:  {args.min_train_window}")
    print(f"  Absent = NO:       {args.absent_means_no}")
    print(f"  Initial capital:   ${args.initial_capital:,.0f}")

    # Ensure ground truth exists
    if not (args.ground_truth_dir / "ground_truth.json").exists():
        print("\nNo ground truth data. Fetching from Kalshi API...")
        dataset = fetch_ground_truth()
        if dataset.contracts:
            save_ground_truth(dataset, args.ground_truth_dir)
        else:
            print("No settled contracts found.")
            return

    # Determine companies
    dataset = load_ground_truth(args.ground_truth_dir)
    if args.company:
        companies = [args.company.upper()]
    else:
        companies = dataset.companies

    # Check which companies have press releases
    pr_dir = Path("data/earnings/press_releases")
    available = []
    for c in companies:
        pr_count = len(list((pr_dir / c).glob(f"{c}_*.txt"))) if (pr_dir / c).exists() else 0
        if pr_count > 0:
            available.append((c, pr_count))
        else:
            print(f"\n  {c}: No press releases found, skipping")

    if not available:
        print("\nNo press releases found for any company.")
        print("Run: python scripts/fetch_real_transcripts.py --mode dates --ticker META")
        return

    # Run expanded and sparse backtests side by side
    print(f"\n{'=' * 70}")
    print("RUNNING BACKTESTS")
    print(f"{'=' * 70}")

    expanded_results = []
    sparse_results = []
    comparison_rows = []

    for ticker, pr_count in available:
        print(f"\n{'─' * 50}")
        print(f"  {ticker} ({pr_count} press releases)")
        print(f"{'─' * 50}")

        # Expanded backtest
        print(f"\n  [EXPANDED] Using press releases + real outcomes:")
        exp_result, exp_diag = run_expanded_backtest(
            ticker=ticker,
            edge_threshold=args.edge_threshold,
            half_life=args.half_life,
            min_train_window=args.min_train_window,
            initial_capital=args.initial_capital,
            absent_means_no=args.absent_means_no,
            ground_truth_dir=args.ground_truth_dir,
        )

        if exp_result:
            expanded_results.append(exp_result)
            m = exp_result.metrics
            print(f"\n    Predictions: {m.total_predictions}")
            print(f"    Accuracy:    {m.accuracy:.1%}")
            print(f"    Brier Score: {m.brier_score:.4f}")
            print(f"    Trades:      {m.total_trades}")
            if m.total_trades > 0:
                print(f"    Win Rate:    {m.win_rate:.1%}")
                print(f"    Total P&L:   ${m.total_pnl:+.2f}")
                print(f"    Sharpe:      {m.sharpe_ratio:.2f}")
        elif exp_diag.get("error"):
            print(f"    SKIPPED: {exp_diag['error']}")

        # Sparse backtest (real-only, for comparison)
        print(f"\n  [SPARSE] Using only real Kalshi outcomes:")
        sp_result, sp_diag = run_sparse_backtest(
            ticker=ticker,
            edge_threshold=args.edge_threshold,
            half_life=args.half_life,
            min_train_window=1,  # sparse needs lower window
            initial_capital=args.initial_capital,
            ground_truth_dir=args.ground_truth_dir,
        )

        if sp_result:
            sparse_results.append(sp_result)
            m = sp_result.metrics
            print(f"    Predictions: {m.total_predictions}")
            print(f"    Accuracy:    {m.accuracy:.1%}")
            print(f"    Trades:      {m.total_trades}")
            if m.total_trades > 0:
                print(f"    Win Rate:    {m.win_rate:.1%}")
                print(f"    Total P&L:   ${m.total_pnl:+.2f}")
        elif sp_diag.get("error"):
            print(f"    SKIPPED: {sp_diag['error']}")

        # Build comparison row
        row = {"ticker": ticker}
        if exp_result:
            row["exp_predictions"] = exp_result.metrics.total_predictions
            row["exp_accuracy"] = exp_result.metrics.accuracy
            row["exp_trades"] = exp_result.metrics.total_trades
            row["exp_pnl"] = exp_result.metrics.total_pnl
            row["exp_brier"] = exp_result.metrics.brier_score
        if sp_result:
            row["sp_predictions"] = sp_result.metrics.total_predictions
            row["sp_accuracy"] = sp_result.metrics.accuracy
            row["sp_trades"] = sp_result.metrics.total_trades
            row["sp_pnl"] = sp_result.metrics.total_pnl
            row["sp_brier"] = sp_result.metrics.brier_score
        comparison_rows.append(row)

    # Print comparison table
    print(f"\n{'=' * 70}")
    print("COMPARISON: EXPANDED vs SPARSE TRAINING DATA")
    print(f"{'=' * 70}")

    if comparison_rows:
        df = pd.DataFrame(comparison_rows).set_index("ticker")
        print(f"\n{'Ticker':<8s} {'--- Expanded ---':>40s}  {'--- Sparse (real only) ---':>40s}")
        print(f"{'':8s} {'Pred':>6s} {'Acc':>6s} {'Brier':>7s} {'Trades':>7s} {'P&L':>9s}  "
              f"{'Pred':>6s} {'Acc':>6s} {'Brier':>7s} {'Trades':>7s} {'P&L':>9s}")
        print(f"{'-' * 100}")

        for _, row in df.iterrows():
            ticker = row.name
            # Expanded
            ep = int(row.get("exp_predictions", 0))
            ea = row.get("exp_accuracy", 0)
            eb = row.get("exp_brier", 0)
            et = int(row.get("exp_trades", 0))
            epnl = row.get("exp_pnl", 0)
            # Sparse
            sp = int(row.get("sp_predictions", 0))
            sa = row.get("sp_accuracy", 0)
            sb = row.get("sp_brier", 0)
            st = int(row.get("sp_trades", 0))
            spnl = row.get("sp_pnl", 0)

            print(
                f"{ticker:<8s} {ep:>6d} {ea:>5.1%} {eb:>7.4f} {et:>7d} ${epnl:>+8.2f}  "
                f"{sp:>6d} {sa:>5.1%} {sb:>7.4f} {st:>7d} ${spnl:>+8.2f}"
            )

        # Totals
        if expanded_results:
            total_exp_preds = sum(r.metrics.total_predictions for r in expanded_results)
            total_exp_trades = sum(r.metrics.total_trades for r in expanded_results)
            total_exp_pnl = sum(r.metrics.total_pnl for r in expanded_results)
            print(f"\n  Expanded total:  {total_exp_preds} predictions, "
                  f"{total_exp_trades} trades, ${total_exp_pnl:+.2f} P&L")

        if sparse_results:
            total_sp_preds = sum(r.metrics.total_predictions for r in sparse_results)
            total_sp_trades = sum(r.metrics.total_trades for r in sparse_results)
            total_sp_pnl = sum(r.metrics.total_pnl for r in sparse_results)
            print(f"  Sparse total:    {total_sp_preds} predictions, "
                  f"{total_sp_trades} trades, ${total_sp_pnl:+.2f} P&L")

    # Save results
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for r in expanded_results:
        company = r.metadata["ticker"]
        save_earnings_backtest_result(r, args.output_dir / company)

    report = {
        "parameters": {
            "edge_threshold": args.edge_threshold,
            "half_life": args.half_life,
            "min_train_window": args.min_train_window,
            "absent_means_no": args.absent_means_no,
        },
        "comparison": comparison_rows,
    }
    report_path = args.output_dir / "comparison_report.json"
    report_path.write_text(json.dumps(report, indent=2, default=str))
    print(f"\n  Report saved to {report_path}")

    print(f"\n{'=' * 70}")
    print("KEY CAVEAT: All results use 0.5 market price baseline.")
    print("Real market prices (from paper trading) will produce much smaller edges.")
    print("Use scripts/paper_trade.py to validate with live prices.")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
