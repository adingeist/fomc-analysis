#!/usr/bin/env python
"""
Batch Training and Parameter Optimization for Earnings Models

This script allows:
1. Training models for multiple tickers in parallel
2. Grid search for optimal parameters
3. Comparison of results across tickers

Usage:
    # Train multiple tickers
    python scripts/batch_train_earnings.py META TSLA NVDA

    # Grid search for one ticker
    python scripts/batch_train_earnings.py META --grid-search

    # Compare all available tickers
    python scripts/batch_train_earnings.py --all-available
"""

import argparse
import asyncio
import json
from datetime import datetime
from itertools import product
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict

import pandas as pd

# Import the trainer
from train_earnings_model import EarningsModelTrainer, TrainingConfig, TrainingResult
from fomc_analysis.kalshi_client_factory import get_kalshi_client, KalshiSdkAdapter


@dataclass
class GridSearchResult:
    """Result from a single grid search trial."""
    params: Dict[str, Any]
    accuracy: float
    roi: float
    sharpe_ratio: float
    total_trades: int
    brier_score: float


@dataclass
class BatchTrainingResult:
    """Results from batch training multiple tickers."""
    tickers: List[str]
    results: Dict[str, TrainingResult]
    comparison_df: pd.DataFrame
    best_ticker: str
    run_timestamp: str


class BatchTrainer:
    """
    Train multiple tickers and optimize parameters.
    """

    def __init__(
        self,
        tickers: List[str],
        output_dir: Path = Path("data/batch_training"),
        num_quarters: int = 12,
        verbose: bool = True,
    ):
        self.tickers = [t.upper() for t in tickers]
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.num_quarters = num_quarters
        self.verbose = verbose

    def log(self, message: str):
        if self.verbose:
            print(message)

    async def discover_available_tickers(self) -> List[str]:
        """Discover all tickers with available Kalshi contracts."""
        self.log("\nDiscovering available tickers on Kalshi...")

        known_tickers = [
            "META", "TSLA", "NVDA", "AMZN", "AAPL", "MSFT", "GOOGL", "NFLX",
            "AMD", "COIN", "UBER", "SNAP", "RBLX", "PLTR", "SHOP", "SQ",
        ]

        try:
            client = get_kalshi_client()
            available = []

            for ticker in known_tickers:
                series_ticker = f"KXEARNINGSMENTION{ticker}"
                try:
                    if isinstance(client, KalshiSdkAdapter):
                        markets = await client.get_markets_async(series_ticker=series_ticker)
                    else:
                        markets = client.get_markets(series_ticker=series_ticker)

                    if markets and len(markets) > 0:
                        available.append(ticker)
                        self.log(f"  [OK] {ticker}: {len(markets)} contracts")
                except Exception:
                    pass

            self.log(f"\nFound {len(available)} tickers with contracts")
            return available

        except Exception as e:
            self.log(f"Error discovering tickers: {e}")
            return []

    async def train_single(
        self,
        ticker: str,
        config_overrides: Dict[str, Any] = None,
    ) -> Optional[TrainingResult]:
        """Train a single ticker with optional config overrides."""
        config_dict = {
            "ticker": ticker,
            "num_quarters": self.num_quarters,
            "use_real_contracts": True,
            "verbose": self.verbose,
            "output_dir": self.output_dir,  # Trainer will create ticker subdirectory
        }

        if config_overrides:
            config_dict.update(config_overrides)

        config = TrainingConfig(**config_dict)
        trainer = EarningsModelTrainer(config)

        try:
            result = await trainer.train()
            return result
        except Exception as e:
            self.log(f"Error training {ticker}: {e}")
            return None

    async def train_all(
        self,
        config_overrides: Dict[str, Any] = None,
    ) -> BatchTrainingResult:
        """Train all tickers concurrently."""
        self.log("\n" + "=" * 60)
        self.log("BATCH TRAINING")
        self.log("=" * 60)
        self.log(f"Tickers: {', '.join(self.tickers)}")
        self.log(f"Quarters: {self.num_quarters}")

        results = {}

        # Train each ticker
        for ticker in self.tickers:
            self.log(f"\n{'='*60}")
            self.log(f"Training {ticker}...")
            self.log("=" * 60)

            result = await self.train_single(ticker, config_overrides)
            if result:
                results[ticker] = result

        # Build comparison dataframe
        comparison_data = []
        for ticker, result in results.items():
            metrics = result.backtest_result.metrics
            comparison_data.append({
                "ticker": ticker,
                "accuracy": metrics.accuracy,
                "brier_score": metrics.brier_score,
                "total_trades": metrics.total_trades,
                "win_rate": metrics.win_rate,
                "total_pnl": metrics.total_pnl,
                "roi": metrics.roi,
                "sharpe_ratio": metrics.sharpe_ratio,
                "contracts_found": result.contracts_found,
                "kalshi_available": result.kalshi_contracts_available,
            })

        comparison_df = pd.DataFrame(comparison_data)

        # Determine best ticker by Sharpe ratio
        if not comparison_df.empty:
            best_idx = comparison_df["sharpe_ratio"].idxmax()
            best_ticker = comparison_df.loc[best_idx, "ticker"]
        else:
            best_ticker = ""

        batch_result = BatchTrainingResult(
            tickers=self.tickers,
            results=results,
            comparison_df=comparison_df,
            best_ticker=best_ticker,
            run_timestamp=datetime.now().strftime("%Y%m%d_%H%M%S"),
        )

        self.print_comparison(batch_result)
        self.save_batch_results(batch_result)

        return batch_result

    async def grid_search(
        self,
        ticker: str,
        param_grid: Dict[str, List[Any]] = None,
    ) -> List[GridSearchResult]:
        """
        Run grid search over parameters for a single ticker.

        Parameters
        ----------
        ticker : str
            Stock ticker to optimize
        param_grid : Dict[str, List[Any]]
            Parameter grid to search. Default searches common parameters.

        Returns
        -------
        List[GridSearchResult]
            Sorted list of results (best first by Sharpe ratio)
        """
        if param_grid is None:
            param_grid = {
                "half_life": [4.0, 6.0, 8.0, 12.0],
                "edge_threshold": [0.08, 0.12, 0.16, 0.20],
                "yes_edge_threshold": [0.15, 0.20, 0.25],
                "no_edge_threshold": [0.05, 0.08, 0.12],
            }

        self.log("\n" + "=" * 60)
        self.log(f"GRID SEARCH: {ticker}")
        self.log("=" * 60)

        # Generate all combinations
        param_names = list(param_grid.keys())
        param_values = list(param_grid.values())
        combinations = list(product(*param_values))

        self.log(f"Parameter grid:")
        for name, values in param_grid.items():
            self.log(f"  {name}: {values}")
        self.log(f"\nTotal combinations: {len(combinations)}")

        results = []
        best_sharpe = float("-inf")
        best_params = None

        for i, combo in enumerate(combinations, 1):
            params = dict(zip(param_names, combo))

            self.log(f"\n[{i}/{len(combinations)}] Testing: {params}")

            config_overrides = {
                "verbose": False,  # Quiet mode for grid search
                **params,
            }

            result = await self.train_single(ticker, config_overrides)

            if result:
                metrics = result.backtest_result.metrics

                grid_result = GridSearchResult(
                    params=params,
                    accuracy=metrics.accuracy,
                    roi=metrics.roi,
                    sharpe_ratio=metrics.sharpe_ratio,
                    total_trades=metrics.total_trades,
                    brier_score=metrics.brier_score,
                )
                results.append(grid_result)

                # Track best
                if metrics.sharpe_ratio > best_sharpe:
                    best_sharpe = metrics.sharpe_ratio
                    best_params = params

                self.log(f"  Accuracy: {metrics.accuracy:.1%}, ROI: {metrics.roi:.1%}, "
                         f"Sharpe: {metrics.sharpe_ratio:.2f}, Trades: {metrics.total_trades}")

        # Sort by Sharpe ratio
        results.sort(key=lambda x: x.sharpe_ratio, reverse=True)

        # Print best results
        self.log("\n" + "=" * 60)
        self.log("GRID SEARCH RESULTS")
        self.log("=" * 60)

        self.log(f"\nBest parameters (by Sharpe ratio):")
        self.log(f"  {best_params}")
        self.log(f"  Sharpe Ratio: {best_sharpe:.2f}")

        self.log(f"\nTop 5 configurations:")
        for i, r in enumerate(results[:5], 1):
            self.log(f"\n{i}. {r.params}")
            self.log(f"   Accuracy: {r.accuracy:.1%}, ROI: {r.roi:.1%}, "
                     f"Sharpe: {r.sharpe_ratio:.2f}, Trades: {r.total_trades}")

        # Save grid search results
        grid_file = self.output_dir / ticker / "grid_search_results.json"
        grid_file.parent.mkdir(parents=True, exist_ok=True)
        with open(grid_file, "w") as f:
            json.dump([asdict(r) for r in results], f, indent=2)

        self.log(f"\nGrid search results saved to {grid_file}")

        return results

    def print_comparison(self, batch_result: BatchTrainingResult):
        """Print comparison table of all tickers."""
        df = batch_result.comparison_df

        if df.empty:
            self.log("\nNo results to compare")
            return

        print("\n" + "=" * 80)
        print("TICKER COMPARISON")
        print("=" * 80)

        # Format for display
        display_df = df.copy()
        display_df["accuracy"] = display_df["accuracy"].apply(lambda x: f"{x:.1%}")
        display_df["win_rate"] = display_df["win_rate"].apply(lambda x: f"{x:.1%}")
        display_df["roi"] = display_df["roi"].apply(lambda x: f"{x:+.1%}")
        display_df["total_pnl"] = display_df["total_pnl"].apply(lambda x: f"${x:,.2f}")
        display_df["sharpe_ratio"] = display_df["sharpe_ratio"].apply(lambda x: f"{x:.2f}")
        display_df["brier_score"] = display_df["brier_score"].apply(lambda x: f"{x:.4f}")

        print("\n" + display_df.to_string(index=False))

        print(f"\nBest Ticker (by Sharpe): {batch_result.best_ticker}")

    def save_batch_results(self, batch_result: BatchTrainingResult):
        """Save batch training results."""
        timestamp = batch_result.run_timestamp

        # Save comparison CSV
        csv_file = self.output_dir / f"comparison_{timestamp}.csv"
        batch_result.comparison_df.to_csv(csv_file, index=False)

        # Save summary JSON
        summary = {
            "tickers": batch_result.tickers,
            "best_ticker": batch_result.best_ticker,
            "run_timestamp": timestamp,
            "summary": batch_result.comparison_df.to_dict(orient="records"),
        }
        json_file = self.output_dir / f"batch_summary_{timestamp}.json"
        with open(json_file, "w") as f:
            json.dump(summary, f, indent=2)

        self.log(f"\nBatch results saved to {self.output_dir}")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Batch train earnings models for multiple tickers",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Train specific tickers
    python scripts/batch_train_earnings.py META TSLA NVDA

    # Train all available tickers
    python scripts/batch_train_earnings.py --all-available

    # Grid search for optimal parameters
    python scripts/batch_train_earnings.py META --grid-search

    # Train with custom parameters
    python scripts/batch_train_earnings.py META TSLA --half-life 6.0 --edge-threshold 0.15
        """,
    )

    parser.add_argument(
        "tickers",
        nargs="*",
        help="Stock ticker symbols to train",
    )

    parser.add_argument(
        "--all-available",
        action="store_true",
        help="Train all tickers with available Kalshi contracts",
    )

    parser.add_argument(
        "--grid-search",
        action="store_true",
        help="Run grid search for parameter optimization (first ticker only)",
    )

    parser.add_argument(
        "--num-quarters", "-n",
        type=int,
        default=12,
        help="Number of quarters for training (default: 12)",
    )

    parser.add_argument(
        "--half-life",
        type=float,
        help="Override half-life parameter for all tickers",
    )

    parser.add_argument(
        "--edge-threshold",
        type=float,
        help="Override edge threshold for all tickers",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/batch_training",
        help="Output directory for results",
    )

    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress verbose output",
    )

    return parser.parse_args()


async def main():
    """Main entry point."""
    args = parse_args()

    tickers = args.tickers or []

    # Create batch trainer
    batch_trainer = BatchTrainer(
        tickers=tickers,
        output_dir=Path(args.output_dir),
        num_quarters=args.num_quarters,
        verbose=not args.quiet,
    )

    # Discover available tickers if requested
    if args.all_available:
        available = await batch_trainer.discover_available_tickers()
        if not available:
            print("No tickers with Kalshi contracts found")
            return
        batch_trainer.tickers = available

    if not batch_trainer.tickers:
        print("No tickers specified. Use --all-available or provide ticker symbols.")
        return

    # Build config overrides
    config_overrides = {}
    if args.half_life:
        config_overrides["half_life"] = args.half_life
    if args.edge_threshold:
        config_overrides["edge_threshold"] = args.edge_threshold

    # Run grid search if requested
    if args.grid_search:
        ticker = batch_trainer.tickers[0]
        await batch_trainer.grid_search(ticker)
    else:
        # Train all tickers
        await batch_trainer.train_all(config_overrides)


if __name__ == "__main__":
    asyncio.run(main())
