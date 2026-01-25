#!/usr/bin/env python
"""
Predict upcoming earnings call word mentions.

This script:
1. Loads trained model for a ticker
2. Fetches current Kalshi contract prices
3. Generates predictions for next earnings call
4. Identifies high-edge trading opportunities

Usage:
    python scripts/predict_earnings.py META
    python scripts/predict_earnings.py META --next-call
    python scripts/predict_earnings.py TSLA --edge-threshold 0.15
    python scripts/predict_earnings.py NVDA --show-all  # Show all predictions

Examples:
    # Basic prediction for META
    python scripts/predict_earnings.py META

    # Show trading recommendations with lower edge threshold
    python scripts/predict_earnings.py META --edge-threshold 0.10

    # Show all predictions including low-edge ones
    python scripts/predict_earnings.py META --show-all
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Dict

import numpy as np
import pandas as pd

from earnings_analysis.models import BetaBinomialEarningsModel
from earnings_analysis.fetchers import (
    KalshiMarketDataFetcher,
    ContractMarketData,
)
from fomc_analysis.kalshi_client_factory import get_kalshi_client


@dataclass
class WordPrediction:
    """Prediction for a single word."""
    word: str
    predicted_probability: float
    confidence_lower: float
    confidence_upper: float
    market_price: float
    edge: float
    recommendation: str  # "BUY YES", "BUY NO", "HOLD"
    edge_quality: str  # "HIGH", "MEDIUM", "LOW"
    historical_mention_rate: float


@dataclass
class EarningsPredictionReport:
    """Complete prediction report for upcoming earnings."""
    ticker: str
    prediction_date: str
    next_call_date: Optional[str]
    predictions: List[WordPrediction]
    top_opportunities: List[WordPrediction]
    model_info: Dict
    market_info: Dict


class EarningsPredictor:
    """
    Generate predictions for upcoming earnings calls.

    Uses:
    - Trained Beta-Binomial model (or trains on historical data)
    - Current Kalshi market prices
    - Historical mention patterns
    """

    def __init__(
        self,
        ticker: str,
        model_dir: Optional[Path] = None,
        yes_edge_threshold: float = 0.22,
        no_edge_threshold: float = 0.08,
    ):
        self.ticker = ticker.upper()
        self.model_dir = model_dir or Path(f"data/training/{self.ticker}")
        self.yes_edge_threshold = yes_edge_threshold
        self.no_edge_threshold = no_edge_threshold

        # Will be populated
        self.model: Optional[BetaBinomialEarningsModel] = None
        self.historical_data: Optional[pd.DataFrame] = None
        self.active_contracts: List[ContractMarketData] = []

    def load_or_train_model(self) -> bool:
        """Load existing model or train new one from historical data."""
        model_file = self.model_dir / "model.json"

        if model_file.exists():
            print(f"Loading trained model from {model_file}")
            self.model = BetaBinomialEarningsModel()
            self.model.load(str(model_file))
            return True

        # Try to load historical data and train
        features_file = self.model_dir / "features.csv"
        outcomes_file = self.model_dir / "outcomes.csv"

        if features_file.exists() and outcomes_file.exists():
            print(f"Training model from historical data in {self.model_dir}")
            features_df = pd.read_csv(features_file, index_col=0)
            outcomes_df = pd.read_csv(outcomes_file, index_col=0)

            self.historical_data = outcomes_df
            return True

        print(f"No model or training data found in {self.model_dir}")
        print("Run train_earnings_model.py first, or use --no-model for market-only analysis")
        return False

    def fetch_active_contracts(self) -> List[ContractMarketData]:
        """Fetch currently active Kalshi contracts."""
        print(f"\nFetching active Kalshi contracts for {self.ticker}...")

        try:
            client = get_kalshi_client()
            fetcher = KalshiMarketDataFetcher(client)

            snapshot = fetcher.fetch_market_data(self.ticker, use_cache=False)
            self.active_contracts = [c for c in snapshot.contracts if c.status == "active"]

            print(f"Found {len(self.active_contracts)} active contracts")
            return self.active_contracts

        except Exception as e:
            print(f"Error fetching contracts: {e}")
            return []

    def predict_word(
        self,
        word: str,
        market_price: float,
        historical_outcomes: Optional[pd.Series] = None,
    ) -> WordPrediction:
        """Generate prediction for a single word."""
        # Train model on historical data for this word
        if historical_outcomes is not None and len(historical_outcomes) > 0:
            model = BetaBinomialEarningsModel(
                alpha_prior=1.0,
                beta_prior=1.0,
                half_life=8.0,
            )
            model.fit(pd.DataFrame(), historical_outcomes)

            pred = model.predict()
            predicted_prob = float(pred.iloc[0]["probability"])
            lower_bound = float(pred.iloc[0]["lower_bound"])
            upper_bound = float(pred.iloc[0]["upper_bound"])
            historical_rate = historical_outcomes.mean()
        else:
            # No historical data - use market price as baseline
            predicted_prob = market_price
            lower_bound = max(0, market_price - 0.2)
            upper_bound = min(1, market_price + 0.2)
            historical_rate = 0.0

        # Calculate edge
        edge = predicted_prob - market_price

        # Determine recommendation
        if edge > 0 and predicted_prob >= 0.65:
            if edge >= self.yes_edge_threshold:
                recommendation = "BUY YES"
                edge_quality = "HIGH" if edge >= self.yes_edge_threshold * 1.5 else "MEDIUM"
            else:
                recommendation = "HOLD"
                edge_quality = "LOW"
        elif edge < 0 and predicted_prob <= 0.35:
            if abs(edge) >= self.no_edge_threshold:
                recommendation = "BUY NO"
                edge_quality = "HIGH" if abs(edge) >= self.no_edge_threshold * 1.5 else "MEDIUM"
            else:
                recommendation = "HOLD"
                edge_quality = "LOW"
        else:
            recommendation = "HOLD"
            edge_quality = "LOW"

        return WordPrediction(
            word=word,
            predicted_probability=predicted_prob,
            confidence_lower=lower_bound,
            confidence_upper=upper_bound,
            market_price=market_price,
            edge=edge,
            recommendation=recommendation,
            edge_quality=edge_quality,
            historical_mention_rate=historical_rate,
        )

    def generate_predictions(self) -> EarningsPredictionReport:
        """Generate predictions for all active contracts."""
        predictions = []

        # Get historical outcomes if available
        outcomes_file = self.model_dir / "outcomes.csv"
        if outcomes_file.exists():
            outcomes_df = pd.read_csv(outcomes_file, index_col=0)
        else:
            outcomes_df = pd.DataFrame()

        for contract in self.active_contracts:
            word = contract.word
            market_price = contract.last_price

            # Get historical data for this word
            historical = None
            if word in outcomes_df.columns:
                historical = outcomes_df[word]
            elif word.lower() in outcomes_df.columns:
                historical = outcomes_df[word.lower()]

            prediction = self.predict_word(word, market_price, historical)
            predictions.append(prediction)

        # Sort by absolute edge
        predictions.sort(key=lambda p: abs(p.edge), reverse=True)

        # Identify top opportunities
        top_opportunities = [
            p for p in predictions
            if p.recommendation in ("BUY YES", "BUY NO")
        ]

        # Get next call date from contracts
        next_call_date = None
        if self.active_contracts:
            dates = [c.call_date for c in self.active_contracts if c.call_date]
            if dates:
                next_call_date = min(dates)

        return EarningsPredictionReport(
            ticker=self.ticker,
            prediction_date=datetime.now().isoformat(),
            next_call_date=next_call_date,
            predictions=predictions,
            top_opportunities=top_opportunities,
            model_info={
                "model_dir": str(self.model_dir),
                "has_trained_model": (self.model_dir / "model.json").exists(),
                "historical_quarters": len(outcomes_df) if not outcomes_df.empty else 0,
            },
            market_info={
                "active_contracts": len(self.active_contracts),
                "words_tracked": len(predictions),
            },
        )

    def print_report(
        self,
        report: EarningsPredictionReport,
        show_all: bool = False,
    ):
        """Print formatted prediction report."""
        print("\n" + "=" * 70)
        print(f"EARNINGS PREDICTION REPORT: {report.ticker}")
        print("=" * 70)
        print(f"\nGenerated: {report.prediction_date}")
        if report.next_call_date:
            print(f"Next Earnings Call: {report.next_call_date}")
        print(f"Active Contracts: {report.market_info['active_contracts']}")
        print(f"Historical Quarters: {report.model_info['historical_quarters']}")

        # Print top opportunities
        if report.top_opportunities:
            print("\n" + "-" * 70)
            print("TOP TRADING OPPORTUNITIES")
            print("-" * 70)
            print(f"{'Word':<25} {'Side':<8} {'Model':>7} {'Market':>7} {'Edge':>8} {'Quality':<8}")
            print("-" * 70)

            for pred in report.top_opportunities[:10]:
                side = "YES" if pred.recommendation == "BUY YES" else "NO"
                print(f"{pred.word:<25} {side:<8} {pred.predicted_probability:>6.1%} "
                      f"{pred.market_price:>6.1%} {pred.edge:>+7.1%} {pred.edge_quality:<8}")
        else:
            print("\n[No high-edge opportunities found]")

        # Print all predictions if requested
        if show_all:
            print("\n" + "-" * 70)
            print("ALL PREDICTIONS")
            print("-" * 70)
            print(f"{'Word':<25} {'Model':>7} {'Market':>7} {'Edge':>8} {'History':>8} {'Action':<10}")
            print("-" * 70)

            for pred in report.predictions:
                print(f"{pred.word:<25} {pred.predicted_probability:>6.1%} "
                      f"{pred.market_price:>6.1%} {pred.edge:>+7.1%} "
                      f"{pred.historical_mention_rate:>7.1%} {pred.recommendation:<10}")

        # Summary statistics
        print("\n" + "-" * 70)
        print("SUMMARY")
        print("-" * 70)

        buy_yes = [p for p in report.predictions if p.recommendation == "BUY YES"]
        buy_no = [p for p in report.predictions if p.recommendation == "BUY NO"]
        hold = [p for p in report.predictions if p.recommendation == "HOLD"]

        print(f"BUY YES signals: {len(buy_yes)}")
        print(f"BUY NO signals: {len(buy_no)}")
        print(f"HOLD signals: {len(hold)}")

        if buy_yes:
            avg_edge_yes = np.mean([p.edge for p in buy_yes])
            print(f"Average YES edge: {avg_edge_yes:+.1%}")

        if buy_no:
            avg_edge_no = np.mean([abs(p.edge) for p in buy_no])
            print(f"Average NO edge: {avg_edge_no:+.1%}")

    def save_report(
        self,
        report: EarningsPredictionReport,
        output_file: Optional[Path] = None,
    ):
        """Save prediction report to JSON."""
        output_file = output_file or (self.model_dir / f"predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        output_file.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "ticker": report.ticker,
            "prediction_date": report.prediction_date,
            "next_call_date": report.next_call_date,
            "predictions": [asdict(p) for p in report.predictions],
            "top_opportunities": [asdict(p) for p in report.top_opportunities],
            "model_info": report.model_info,
            "market_info": report.market_info,
        }

        with open(output_file, "w") as f:
            json.dump(data, f, indent=2)

        print(f"\nReport saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Predict upcoming earnings call word mentions",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "ticker",
        type=str,
        help="Stock ticker symbol (e.g., META, TSLA, NVDA)",
    )

    parser.add_argument(
        "--model-dir",
        type=Path,
        help="Directory containing trained model (default: data/training/{ticker})",
    )

    parser.add_argument(
        "--yes-edge-threshold",
        type=float,
        default=0.22,
        help="Minimum edge for YES recommendations (default: 0.22)",
    )

    parser.add_argument(
        "--no-edge-threshold",
        type=float,
        default=0.08,
        help="Minimum edge for NO recommendations (default: 0.08)",
    )

    parser.add_argument(
        "--edge-threshold",
        type=float,
        help="Set both YES and NO thresholds (overrides individual thresholds)",
    )

    parser.add_argument(
        "--show-all",
        action="store_true",
        help="Show all predictions, not just opportunities",
    )

    parser.add_argument(
        "--no-model",
        action="store_true",
        help="Skip model loading, use market-only analysis",
    )

    parser.add_argument(
        "--output",
        type=Path,
        help="Save report to this file",
    )

    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Only print top opportunities",
    )

    args = parser.parse_args()

    # Handle edge threshold shortcut
    yes_threshold = args.yes_edge_threshold
    no_threshold = args.no_edge_threshold
    if args.edge_threshold is not None:
        yes_threshold = args.edge_threshold
        no_threshold = args.edge_threshold

    # Create predictor
    predictor = EarningsPredictor(
        ticker=args.ticker,
        model_dir=args.model_dir,
        yes_edge_threshold=yes_threshold,
        no_edge_threshold=no_threshold,
    )

    # Load model (optional)
    if not args.no_model:
        predictor.load_or_train_model()

    # Fetch active contracts
    contracts = predictor.fetch_active_contracts()

    if not contracts:
        print(f"\nNo active Kalshi contracts found for {args.ticker}")
        print("This ticker may not have earnings mention contracts, or contracts may have expired.")
        sys.exit(1)

    # Generate predictions
    report = predictor.generate_predictions()

    # Print report
    predictor.print_report(report, show_all=args.show_all)

    # Save if requested
    if args.output:
        predictor.save_report(report, args.output)
    elif not args.quiet:
        # Auto-save to default location
        predictor.save_report(report)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nPrediction interrupted")
        sys.exit(1)
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
