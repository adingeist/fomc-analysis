#!/usr/bin/env python3
"""
Analyze earnings contract microstructure data.

Usage:
    python scripts/analyze_earnings_microstructure.py [--company META] [--all]

Analyses:
    --calibration     Build earnings-specific calibration curve
    --spreads         Analyze bid-ask spreads
    --efficiency      Compute market efficiency scores
    --opportunities   Find new contract opportunities
    --all             Run all analyses

Examples:
    # Full analysis for all companies
    python scripts/analyze_earnings_microstructure.py --all

    # Just efficiency monitoring
    python scripts/analyze_earnings_microstructure.py --efficiency

    # Calibration for specific company
    python scripts/analyze_earnings_microstructure.py --calibration --company META
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from earnings_analysis.microstructure.trade_storage import ParquetStorage
from earnings_analysis.microstructure.trade_analyzer import EarningsTradeAnalyzer
from earnings_analysis.microstructure.regime import (
    EfficiencyMonitor,
    AdaptiveThresholds,
    NewContractDetector,
)
from earnings_analysis.microstructure.calibration import KalshiCalibrationCurve


def run_calibration(analyzer, companies, output_dir):
    """Build earnings-specific calibration curve."""
    print("\n=== Building Earnings Calibration Curve ===")

    cal_data = analyzer.build_earnings_calibration(companies)
    if not cal_data:
        print("  No finalized market data available for calibration.")
        return

    print(f"  Calibration data for {len(cal_data)} price levels")

    # Load into calibration curve
    curve = KalshiCalibrationCurve()
    curve.load_empirical_data(cal_data)

    # Compare with parametric model
    print(f"\n  {'Price':>5}  {'Parametric':>10}  {'Empirical':>10}  {'Difference':>10}")
    print(f"  {'-'*5}  {'-'*10}  {'-'*10}  {'-'*10}")

    default_curve = KalshiCalibrationCurve()
    for cents, emp_rate in sorted(cal_data.items()):
        param_rate = default_curve.calibrated_probability(cents)
        diff = emp_rate - param_rate
        print(f"  {cents:>5}  {param_rate:>10.4f}  {emp_rate:>10.4f}  {diff:>+10.4f}")

    # Save calibration data
    output_path = output_dir / "earnings_calibration.json"
    output_path.write_text(json.dumps(cal_data, indent=2))
    print(f"\n  Saved to {output_path}")


def run_spreads(analyzer, companies, output_dir):
    """Analyze bid-ask spreads."""
    print("\n=== Spread Analysis ===")

    for company in companies:
        series_ticker = f"KXEARNINGSMENTION{company}"
        try:
            df = analyzer.spread_analysis(series_ticker)
            if df.empty:
                continue
            print(f"\n  {company}:")
            print(df.to_string(index=False))
        except Exception as e:
            print(f"  {company}: {e}")


def run_efficiency(storage, companies, output_dir):
    """Compute market efficiency scores."""
    print("\n=== Market Efficiency Monitoring ===")

    monitor = EfficiencyMonitor(storage=storage)
    metrics = monitor.monitor_all(companies)

    if not metrics:
        print("  No data available.")
        return

    thresholds = AdaptiveThresholds()
    all_thresholds = thresholds.compute_all_thresholds(metrics)

    print(f"\n  {'Company':>8}  {'Score':>6}  {'Spread':>7}  {'Volume':>7}  {'YES Thresh':>10}  {'NO Thresh':>10}")
    print(f"  {'-'*8}  {'-'*6}  {'-'*7}  {'-'*7}  {'-'*10}  {'-'*10}")

    for m in metrics:
        t = all_thresholds[m.company]
        print(
            f"  {m.company:>8}  {m.efficiency_score:>6.3f}  "
            f"{m.avg_spread_cents:>7.1f}  {m.avg_volume:>7.0f}  "
            f"{t['yes_threshold']:>10.3f}  {t['no_threshold']:>10.3f}"
        )

    # Save
    output_path = output_dir / "efficiency_report.json"
    report = {
        "metrics": [m.to_dict() for m in metrics],
        "thresholds": all_thresholds,
    }
    output_path.write_text(json.dumps(report, indent=2, default=str))
    print(f"\n  Saved to {output_path}")


def run_opportunities(storage, companies, output_dir):
    """Find new contract opportunities."""
    print("\n=== New Contract Opportunities ===")

    detector = NewContractDetector(storage=storage)
    summary = detector.summarize_opportunities(companies)

    if summary["total_opportunities"] == 0:
        print("  No opportunities found.")
        return

    print(f"  Total opportunities: {summary['total_opportunities']}")
    print(f"  Avg spread: {summary['avg_spread']:.1f}c")
    print(f"  Avg opportunity score: {summary['avg_opportunity_score']:.3f}")

    if summary.get("by_company"):
        print(f"\n  By company:")
        for company, count in sorted(summary["by_company"].items()):
            print(f"    {company}: {count}")

    print(f"\n  Top 10 contracts:")
    print(f"  {'Ticker':<45}  {'Word':<20}  {'Spread':>6}  {'Score':>6}")
    print(f"  {'-'*45}  {'-'*20}  {'-'*6}  {'-'*6}")
    for c in summary["top_contracts"]:
        print(
            f"  {c['ticker']:<45}  {c['word']:<20}  "
            f"{c['spread_cents']:>6}  {c['opportunity_score']:>6.3f}"
        )

    # Save
    output_path = output_dir / "opportunities.json"
    output_path.write_text(json.dumps(summary, indent=2, default=str))
    print(f"\n  Saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze earnings contract microstructure"
    )
    parser.add_argument("--company", type=str, help="Single company ticker")
    parser.add_argument("--calibration", action="store_true")
    parser.add_argument("--spreads", action="store_true")
    parser.add_argument("--efficiency", action="store_true")
    parser.add_argument("--opportunities", action="store_true")
    parser.add_argument("--all", action="store_true", help="Run all analyses")
    parser.add_argument(
        "--data-dir", type=str, default="data/microstructure",
        help="Data directory",
    )

    args = parser.parse_args()

    # Default to --all if nothing specified
    if not any([args.calibration, args.spreads, args.efficiency, args.opportunities]):
        args.all = True

    companies = None
    if args.company:
        companies = [args.company.upper()]

    storage = ParquetStorage(data_dir=Path(args.data_dir))
    output_dir = Path(args.data_dir) / "reports"
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.all or args.calibration:
        try:
            analyzer = EarningsTradeAnalyzer(storage=storage)
            run_calibration(analyzer, companies, output_dir)
            if args.all or args.spreads:
                run_spreads(
                    analyzer,
                    companies or ["META", "TSLA", "NVDA", "AAPL", "MSFT", "AMZN", "GOOGL", "NFLX"],
                    output_dir,
                )
            analyzer.close()
        except ImportError as e:
            print(f"  DuckDB not available: {e}")

    if args.all or args.efficiency:
        run_efficiency(
            storage,
            companies or ["META", "TSLA", "NVDA", "AAPL", "MSFT", "AMZN", "GOOGL", "NFLX"],
            output_dir,
        )

    if args.all or args.opportunities:
        run_opportunities(
            storage,
            companies,
            output_dir,
        )

    print("\n=== Analysis complete ===")


if __name__ == "__main__":
    main()
