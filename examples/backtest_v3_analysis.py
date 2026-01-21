"""
Advanced analysis of backtest v3 results.

This script demonstrates how to:
1. Load backtest results
2. Analyze prediction accuracy by contract
3. Identify most profitable contracts
4. Compare performance across time horizons
5. Generate visualizations (optional, requires matplotlib)
"""

import json
from pathlib import Path
from typing import Dict, List

import pandas as pd
import numpy as np


def load_backtest_results(results_dir: Path) -> Dict:
    """Load backtest results from JSON file."""
    results_file = results_dir / "backtest_results.json"
    return json.loads(results_file.read_text())


def analyze_contract_accuracy(predictions_df: pd.DataFrame) -> pd.DataFrame:
    """Analyze prediction accuracy by contract."""
    contract_stats = predictions_df.groupby('contract').agg({
        'correct': ['mean', 'sum', 'count'],
        'predicted_probability': 'mean',
        'actual_outcome': 'mean',
    }).round(3)

    contract_stats.columns = ['accuracy', 'correct_count', 'total_predictions',
                               'avg_predicted_prob', 'actual_yes_rate']
    contract_stats = contract_stats.sort_values('accuracy', ascending=False)

    return contract_stats


def analyze_contract_profitability(trades_df: pd.DataFrame) -> pd.DataFrame:
    """Analyze profitability by contract."""
    contract_pnl = trades_df.groupby('contract').agg({
        'pnl': ['sum', 'mean', 'count'],
        'roi': 'mean',
        'edge': 'mean',
    }).round(2)

    contract_pnl.columns = ['total_pnl', 'avg_pnl', 'num_trades', 'avg_roi', 'avg_edge']
    contract_pnl = contract_pnl.sort_values('total_pnl', ascending=False)

    return contract_pnl


def analyze_horizon_performance(predictions_df: pd.DataFrame, trades_df: pd.DataFrame) -> pd.DataFrame:
    """Compare performance across time horizons."""
    horizon_stats = []

    for horizon in sorted(predictions_df['days_before_meeting'].unique()):
        horizon_preds = predictions_df[predictions_df['days_before_meeting'] == horizon]
        horizon_trades = trades_df[trades_df['days_before_meeting'] == horizon]

        accuracy = horizon_preds['correct'].mean()
        brier_score = ((horizon_preds['predicted_probability'] - horizon_preds['actual_outcome']) ** 2).mean()

        if len(horizon_trades) > 0:
            total_pnl = horizon_trades['pnl'].sum()
            avg_pnl = horizon_trades['pnl'].mean()
            win_rate = (horizon_trades['pnl'] > 0).mean()
            sharpe = horizon_trades['roi'].mean() / horizon_trades['roi'].std() if horizon_trades['roi'].std() > 0 else 0
        else:
            total_pnl = avg_pnl = win_rate = sharpe = 0

        horizon_stats.append({
            'days_before': horizon,
            'predictions': len(horizon_preds),
            'accuracy': accuracy,
            'brier_score': brier_score,
            'trades': len(horizon_trades),
            'win_rate': win_rate,
            'total_pnl': total_pnl,
            'avg_pnl': avg_pnl,
            'sharpe': sharpe,
        })

    return pd.DataFrame(horizon_stats)


def analyze_edge_calibration(predictions_df: pd.DataFrame) -> pd.DataFrame:
    """Analyze how edge size relates to prediction accuracy."""
    # Only look at predictions with market prices
    with_prices = predictions_df[predictions_df['market_price'].notna()].copy()
    with_prices['abs_edge'] = with_prices['edge'].abs()

    # Bin by edge size
    edge_bins = [0, 0.05, 0.10, 0.15, 0.20, 1.0]
    edge_labels = ['0-5%', '5-10%', '10-15%', '15-20%', '20%+']
    with_prices['edge_bin'] = pd.cut(with_prices['abs_edge'], bins=edge_bins, labels=edge_labels)

    edge_calibration = with_prices.groupby('edge_bin').agg({
        'correct': ['mean', 'count'],
        'abs_edge': 'mean',
    }).round(3)

    edge_calibration.columns = ['accuracy', 'count', 'avg_abs_edge']

    return edge_calibration


def print_summary(results: Dict):
    """Print a comprehensive summary of backtest results."""
    print("=" * 80)
    print("BACKTEST V3 RESULTS SUMMARY")
    print("=" * 80)

    # Overall metrics
    overall = results['overall_metrics']
    print("\nOVERALL PERFORMANCE:")
    print(f"  Total trades: {overall['total_trades']}")
    print(f"  Total P&L: ${overall['total_pnl']:,.2f}")
    print(f"  ROI: {overall['roi'] * 100:.2f}%")
    print(f"  Win rate: {overall['win_rate'] * 100:.1f}%")
    print(f"  Sharpe ratio: {overall['sharpe']:.2f}")
    print(f"  Avg P&L per trade: ${overall['avg_pnl_per_trade']:,.2f}")
    print(f"  Final capital: ${overall['final_capital']:,.2f}")

    # Horizon breakdown
    print("\nPERFORMANCE BY TIME HORIZON:")
    for horizon, metrics in sorted(results['horizon_metrics'].items(), key=lambda x: int(x[0])):
        print(f"\n  {horizon} days before meeting:")
        print(f"    Predictions: {metrics['total_predictions']}")
        print(f"    Accuracy: {metrics['accuracy'] * 100:.1f}%")
        print(f"    Brier score: {metrics['brier_score']:.3f}")
        print(f"    Trades: {metrics['total_trades']}")
        print(f"    Win rate: {metrics['win_rate'] * 100:.1f}%")
        print(f"    Total P&L: ${metrics['total_pnl']:,.2f}")
        print(f"    ROI: {metrics['roi'] * 100:.1f}%")

    # Model info
    metadata = results['metadata']
    print("\nMODEL CONFIGURATION:")
    print(f"  Model: {metadata['model_class']}")
    print(f"  Parameters: {metadata['model_params']}")
    print(f"  Edge threshold: {metadata['edge_threshold'] * 100:.1f}%")
    print(f"  Position size: {metadata['position_size_pct'] * 100:.1f}%")
    print(f"  Fee rate: {metadata['fee_rate'] * 100:.1f}%")


def main():
    """Run comprehensive backtest analysis."""
    # Load results
    results_dir = Path("results/backtest_v3")

    if not results_dir.exists():
        print(f"Error: Results directory not found: {results_dir}")
        print("Run the backtest first using: fomc-analysis backtest-v3 ...")
        return

    print("Loading backtest results...")
    results = load_backtest_results(results_dir)

    # Print summary
    print_summary(results)

    # Load dataframes
    predictions_df = pd.DataFrame(results['predictions'])
    trades_df = pd.DataFrame(results['trades'])

    # Detailed analyses
    print("\n" + "=" * 80)
    print("DETAILED ANALYSIS")
    print("=" * 80)

    # Contract accuracy
    print("\nTOP 10 MOST ACCURATE CONTRACTS:")
    contract_accuracy = analyze_contract_accuracy(predictions_df)
    print(contract_accuracy.head(10).to_string())

    # Contract profitability
    if len(trades_df) > 0:
        print("\nTOP 10 MOST PROFITABLE CONTRACTS:")
        contract_pnl = analyze_contract_profitability(trades_df)
        print(contract_pnl.head(10).to_string())

        # Horizon performance
        print("\nHORIZON PERFORMANCE COMPARISON:")
        horizon_perf = analyze_horizon_performance(predictions_df, trades_df)
        print(horizon_perf.to_string(index=False))

        # Edge calibration
        print("\nEDGE SIZE vs ACCURACY:")
        edge_cal = analyze_edge_calibration(predictions_df)
        print(edge_cal.to_string())

    # Save detailed analysis
    output_file = results_dir / "detailed_analysis.txt"
    print(f"\nSaving detailed analysis to: {output_file}")

    with output_file.open('w') as f:
        f.write("BACKTEST V3 DETAILED ANALYSIS\n")
        f.write("=" * 80 + "\n\n")

        f.write("CONTRACT ACCURACY:\n")
        f.write(contract_accuracy.to_string() + "\n\n")

        if len(trades_df) > 0:
            f.write("CONTRACT PROFITABILITY:\n")
            f.write(contract_pnl.to_string() + "\n\n")

            f.write("HORIZON PERFORMANCE:\n")
            f.write(horizon_perf.to_string(index=False) + "\n\n")

            f.write("EDGE CALIBRATION:\n")
            f.write(edge_cal.to_string() + "\n\n")

    print("\n" + "=" * 80)
    print("Analysis complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
