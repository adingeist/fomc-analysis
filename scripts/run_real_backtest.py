#!/usr/bin/env python3
"""
Run a real backtest using actual Kalshi ground truth data.

This is the critical validation step: take real finalized contract outcomes,
run walk-forward backtesting, and determine whether the edge is genuinely
positive on out-of-sample data.

Prerequisites:
    python scripts/build_ground_truth.py   # builds data/ground_truth/

Usage:
    python scripts/run_real_backtest.py
    python scripts/run_real_backtest.py --company META
    python scripts/run_real_backtest.py --company NVDA --edge-threshold 0.10
    python scripts/run_real_backtest.py --all-companies
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from earnings_analysis.ground_truth import (
    fetch_ground_truth,
    save_ground_truth,
    load_ground_truth,
    build_backtest_dataframes,
    GroundTruthDataset,
)
from earnings_analysis.models.beta_binomial import BetaBinomialEarningsModel
from earnings_analysis.kalshi.backtester import (
    EarningsKalshiBacktester,
    BacktestResult,
    save_earnings_backtest_result,
    create_microstructure_backtester,
)


def run_backtest_for_company(
    dataset: GroundTruthDataset,
    company: str,
    edge_threshold: float = 0.12,
    half_life: float = 8.0,
    min_train_window: int = 3,
    use_microstructure: bool = True,
    initial_capital: float = 10000.0,
) -> tuple[BacktestResult | None, dict]:
    """Run backtest for a single company using real data.

    Returns (result, diagnostics) tuple.
    """
    features, outcomes, market_prices = build_backtest_dataframes(dataset, company)

    diagnostics = {
        "company": company,
        "num_events": len(outcomes) if not outcomes.empty else 0,
        "num_words": len(outcomes.columns) if not outcomes.empty else 0,
        "date_range": None,
        "words": [],
        "error": None,
    }

    if outcomes.empty:
        diagnostics["error"] = "No outcome data available"
        return None, diagnostics

    diagnostics["num_events"] = len(outcomes)
    diagnostics["num_words"] = len(outcomes.columns)
    diagnostics["date_range"] = f"{outcomes.index.min()} to {outcomes.index.max()}"
    diagnostics["words"] = list(outcomes.columns)

    if len(outcomes) < min_train_window + 1:
        diagnostics["error"] = (
            f"Need at least {min_train_window + 1} events for walk-forward, "
            f"only have {len(outcomes)}"
        )
        return None, diagnostics

    print(f"\n  Running backtest for {company}:")
    print(f"    Events: {len(outcomes)}")
    print(f"    Words: {list(outcomes.columns)[:8]}{'...' if len(outcomes.columns) > 8 else ''}")
    print(f"    Date range: {outcomes.index.min().date()} to {outcomes.index.max().date()}")

    model_params = {
        "alpha_prior": 1.0,
        "beta_prior": 1.0,
        "half_life": half_life,
    }

    # With sparse real data (few events), allow predictions from the prior
    # even when training outcomes have no variation yet.
    if use_microstructure:
        backtester = create_microstructure_backtester(
            features=features,
            outcomes=outcomes,
            model_class=BetaBinomialEarningsModel,
            model_params=model_params,
            edge_threshold=edge_threshold,
            min_train_window=min_train_window,
            require_variation=False,
        )
    else:
        backtester = EarningsKalshiBacktester(
            features=features,
            outcomes=outcomes,
            model_class=BetaBinomialEarningsModel,
            model_params=model_params,
            edge_threshold=edge_threshold,
            min_train_window=min_train_window,
            require_variation=False,
        )

    # We don't have pre-settlement market prices, so pass None to use the
    # 0.5 baseline.  Settlement prices (0/1) would leak the actual outcome.
    result = backtester.run(
        ticker=company,
        initial_capital=initial_capital,
        market_prices=None,
    )

    return result, diagnostics


def print_result(result: BacktestResult, diagnostics: dict):
    """Print backtest results."""
    company = diagnostics["company"]
    m = result.metrics

    print(f"\n  Results for {company}:")
    print(f"    Predictions:    {m.total_predictions}")
    print(f"    Accuracy:       {m.accuracy:.1%}")
    print(f"    Brier Score:    {m.brier_score:.4f}")
    print(f"    Trades:         {m.total_trades}")

    if m.total_trades > 0:
        print(f"    Win Rate:       {m.win_rate:.1%}")
        print(f"    Total P&L:      ${m.total_pnl:+.2f}")
        print(f"    Avg P&L/Trade:  ${m.avg_pnl_per_trade:+.2f}")
        print(f"    ROI:            {m.roi:+.1%}")
        print(f"    Sharpe Ratio:   {m.sharpe_ratio:.2f}")

        # Break down by YES/NO
        yes_trades = [t for t in result.trades if t.side == "YES"]
        no_trades = [t for t in result.trades if t.side == "NO"]
        if yes_trades:
            yes_pnl = sum(t.pnl for t in yes_trades)
            yes_wins = sum(1 for t in yes_trades if t.pnl > 0)
            print(f"    YES trades:     {len(yes_trades)} ({yes_wins} wins, ${yes_pnl:+.2f})")
        if no_trades:
            no_pnl = sum(t.pnl for t in no_trades)
            no_wins = sum(1 for t in no_trades if t.pnl > 0)
            print(f"    NO trades:      {len(no_trades)} ({no_wins} wins, ${no_pnl:+.2f})")
    else:
        print("    (No trades triggered - edge threshold may be too high)")


def assess_edge_significance(results: list[BacktestResult]) -> dict:
    """Assess whether the combined edge across companies is statistically significant."""
    all_trades = []
    for r in results:
        all_trades.extend(r.trades)

    if not all_trades:
        return {"significant": False, "reason": "No trades executed"}

    returns = [t.roi for t in all_trades]
    n = len(returns)
    mean_return = np.mean(returns)
    std_return = np.std(returns, ddof=1)

    # t-test: H0: mean_return = 0
    if std_return > 0 and n > 1:
        t_stat = mean_return / (std_return / np.sqrt(n))
        # Approximate p-value using normal approximation for large n
        from scipy.stats import t as t_dist
        p_value = 2 * (1 - t_dist.cdf(abs(t_stat), df=n - 1))
    else:
        t_stat = 0
        p_value = 1.0

    # Win rate binomial test
    wins = sum(1 for r in returns if r > 0)
    from scipy.stats import binomtest
    try:
        win_p_value = binomtest(wins, n, 0.5, alternative="greater").pvalue
    except Exception:
        win_p_value = 1.0

    return {
        "total_trades": n,
        "mean_return": float(mean_return),
        "std_return": float(std_return),
        "t_statistic": float(t_stat),
        "p_value": float(p_value),
        "significant_at_5pct": p_value < 0.05,
        "significant_at_10pct": p_value < 0.10,
        "win_rate": wins / n if n > 0 else 0,
        "win_rate_p_value": float(win_p_value),
        "total_pnl": float(sum(t.pnl for t in all_trades)),
        "sharpe": float(mean_return / std_return * np.sqrt(n)) if std_return > 0 else 0,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Run real backtest using Kalshi ground truth data"
    )
    parser.add_argument("--company", type=str, help="Single company ticker (e.g. META)")
    parser.add_argument("--all-companies", action="store_true", help="Backtest all companies")
    parser.add_argument("--edge-threshold", type=float, default=0.12)
    parser.add_argument("--half-life", type=float, default=8.0)
    parser.add_argument("--min-train-window", type=int, default=3)
    parser.add_argument("--no-microstructure", action="store_true")
    parser.add_argument("--initial-capital", type=float, default=10000.0)
    parser.add_argument("--ground-truth-dir", type=Path, default=Path("data/ground_truth"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/real_backtest"))
    parser.add_argument(
        "--fetch-fresh",
        action="store_true",
        help="Fetch fresh ground truth from API instead of loading from disk",
    )
    args = parser.parse_args()

    print("=" * 70)
    print("REAL DATA BACKTEST - KALSHI EARNINGS MENTION CONTRACTS")
    print("=" * 70)
    print(f"\nParameters:")
    print(f"  Edge threshold:    {args.edge_threshold}")
    print(f"  Half-life:         {args.half_life}")
    print(f"  Min train window:  {args.min_train_window}")
    print(f"  Microstructure:    {'disabled' if args.no_microstructure else 'enabled'}")
    print(f"  Initial capital:   ${args.initial_capital:,.0f}")

    # Load or fetch ground truth
    if args.fetch_fresh or not (args.ground_truth_dir / "ground_truth.json").exists():
        print("\n[1/4] Fetching ground truth from Kalshi API...")
        dataset = fetch_ground_truth()
        if dataset.contracts:
            save_ground_truth(dataset, args.ground_truth_dir)
        else:
            print("No settled contracts found. Cannot run backtest.")
            return
    else:
        print(f"\n[1/4] Loading ground truth from {args.ground_truth_dir}...")
        dataset = load_ground_truth(args.ground_truth_dir)
        print(f"  Loaded {len(dataset.contracts)} contracts")

    summary = dataset.summary()
    print(f"  Companies: {', '.join(summary['companies'])}")
    print(f"  Total contracts: {summary['total_contracts']}")

    # Determine which companies to backtest
    if args.company:
        companies = [args.company.upper()]
    elif args.all_companies:
        companies = summary["companies"]
    else:
        # Default: all companies with enough data
        companies = summary["companies"]

    # Run backtests
    print(f"\n[2/4] Running walk-forward backtests...")
    all_results = []
    all_diagnostics = []

    for company in companies:
        result, diagnostics = run_backtest_for_company(
            dataset=dataset,
            company=company,
            edge_threshold=args.edge_threshold,
            half_life=args.half_life,
            min_train_window=args.min_train_window,
            use_microstructure=not args.no_microstructure,
            initial_capital=args.initial_capital,
        )

        all_diagnostics.append(diagnostics)

        if result:
            all_results.append(result)
            print_result(result, diagnostics)
        elif diagnostics.get("error"):
            print(f"\n  {company}: SKIPPED - {diagnostics['error']}")

    # Aggregate results
    print(f"\n[3/4] Aggregate Results")
    print("=" * 50)

    if not all_results:
        print("No backtests completed. Possible reasons:")
        print("  - Not enough settled contracts for walk-forward validation")
        print("  - No companies have enough event dates")
        print("  - Try lowering --min-train-window (currently {})".format(args.min_train_window))
        return

    total_trades = sum(r.metrics.total_trades for r in all_results)
    total_pnl = sum(r.metrics.total_pnl for r in all_results)
    total_predictions = sum(r.metrics.total_predictions for r in all_results)

    print(f"  Companies tested:   {len(all_results)}")
    print(f"  Total predictions:  {total_predictions}")
    print(f"  Total trades:       {total_trades}")
    print(f"  Combined P&L:       ${total_pnl:+.2f}")

    # Statistical significance
    if total_trades > 0:
        print(f"\n  Edge Significance Test:")
        sig = assess_edge_significance(all_results)
        print(f"    Mean return:      {sig['mean_return']:+.4f}")
        print(f"    Std return:       {sig['std_return']:.4f}")
        print(f"    t-statistic:      {sig['t_statistic']:.3f}")
        print(f"    p-value:          {sig['p_value']:.4f}")
        print(f"    Win rate:         {sig['win_rate']:.1%}")
        print(f"    Win rate p-value: {sig['win_rate_p_value']:.4f}")
        print(f"    Sharpe ratio:     {sig['sharpe']:.2f}")

        if sig["significant_at_5pct"]:
            print(f"    SIGNIFICANT at 5% level")
        elif sig["significant_at_10pct"]:
            print(f"    MARGINALLY SIGNIFICANT at 10% level")
        else:
            print(f"    NOT SIGNIFICANT at 10% level")
            print(f"    The edge is likely noise or requires more data.")

    # Save results
    print(f"\n[4/4] Saving results to {args.output_dir}...")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for result in all_results:
        company = result.metadata["ticker"]
        company_dir = args.output_dir / company
        save_earnings_backtest_result(result, company_dir)

    # Save aggregate report
    report = {
        "parameters": {
            "edge_threshold": args.edge_threshold,
            "half_life": args.half_life,
            "min_train_window": args.min_train_window,
            "microstructure": not args.no_microstructure,
            "initial_capital": args.initial_capital,
        },
        "ground_truth_summary": summary,
        "companies_tested": [d["company"] for d in all_diagnostics],
        "companies_with_results": [r.metadata["ticker"] for r in all_results],
        "aggregate": {
            "total_predictions": total_predictions,
            "total_trades": total_trades,
            "combined_pnl": total_pnl,
        },
        "diagnostics": all_diagnostics,
    }

    if total_trades > 0:
        report["significance"] = assess_edge_significance(all_results)

    report_path = args.output_dir / "backtest_report.json"
    report_path.write_text(json.dumps(report, indent=2, default=str))
    print(f"  Report saved to {report_path}")

    print("\nReal data backtest complete.")
    if total_trades == 0:
        print(
            "\nNote: Zero trades were triggered. This means the model's edge "
            "never exceeded the threshold. Consider:\n"
            "  - Lowering --edge-threshold\n"
            "  - The market may be well-calibrated for these contracts\n"
            "  - More data may be needed"
        )


if __name__ == "__main__":
    main()
