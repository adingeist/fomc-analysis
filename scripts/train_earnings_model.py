#!/usr/bin/env python
"""
Train Earnings Call Analysis Model - Universal Ticker Support

This script provides a unified entry point for training and backtesting
earnings call prediction models for any ticker.

Usage:
    python scripts/train_earnings_model.py AAPL
    python scripts/train_earnings_model.py TSLA --num-quarters 12
    python scripts/train_earnings_model.py META --use-real-transcripts
    python scripts/train_earnings_model.py NVDA --edge-threshold 0.15 --half-life 6.0

Features:
    - Validates Kalshi contracts exist for the ticker
    - Fetches real transcripts or generates mock data for testing
    - Runs walk-forward backtesting with Beta-Binomial model
    - Outputs comprehensive results including P&L, accuracy, and trade logs
"""

import argparse
import asyncio
import json
import re
import sys
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional, Dict, Any

import numpy as np
import pandas as pd

# Project imports
from earnings_analysis.kalshi import EarningsContractWord, EarningsMentionAnalysis
from earnings_analysis.kalshi.backtester import (
    EarningsKalshiBacktester,
    BacktestResult,
    save_earnings_backtest_result,
)
from earnings_analysis.kalshi.enhanced_backtester import (
    EnhancedEarningsBacktester,
    save_enhanced_backtest_result,
)
from earnings_analysis.models import BetaBinomialEarningsModel, FeatureAwareEarningsModel
from earnings_analysis.features.market_features import MarketFeatureExtractor
from fomc_analysis.kalshi_client_factory import get_kalshi_client, KalshiSdkAdapter


@dataclass
class TrainingConfig:
    """Configuration for training run."""
    ticker: str
    num_quarters: int = 12
    use_real_transcripts: bool = False
    use_real_contracts: bool = True
    use_enhanced_backtester: bool = False  # Use Kelly criterion & advanced features
    use_market_features: bool = False  # Use stock/earnings features

    # Model parameters
    alpha_prior: float = 1.0
    beta_prior: float = 1.0
    half_life: float = 8.0
    market_weight: float = 0.3  # Weight for market price signal

    # Trading parameters (basic backtester)
    edge_threshold: float = 0.12
    yes_edge_threshold: float = 0.22
    no_edge_threshold: float = 0.08
    position_size_pct: float = 0.03
    fee_rate: float = 0.07
    min_train_window: int = 4
    initial_capital: float = 10000.0

    # Enhanced backtester parameters
    kelly_fraction: float = 0.25  # Fraction of Kelly to bet (quarter Kelly)
    max_position_pct: float = 0.10  # Max position size
    confidence_scaling: bool = True  # Scale by model confidence
    correlation_limit: float = 0.25  # Max correlated exposure
    max_drawdown_limit: float = 0.20  # Stop if drawdown exceeds

    # Output
    output_dir: Path = Path("data/training")
    verbose: bool = True


@dataclass
class TrainingResult:
    """Complete training run result."""
    ticker: str
    config: TrainingConfig
    contracts_found: int
    words_tracked: List[str]
    backtest_result: BacktestResult
    run_timestamp: str
    kalshi_contracts_available: bool

    def to_dict(self) -> dict:
        return {
            "ticker": self.ticker,
            "config": asdict(self.config),
            "contracts_found": self.contracts_found,
            "words_tracked": self.words_tracked,
            "kalshi_contracts_available": self.kalshi_contracts_available,
            "run_timestamp": self.run_timestamp,
            "metrics": asdict(self.backtest_result.metrics),
            "metadata": self.backtest_result.metadata,
        }


class EarningsModelTrainer:
    """
    Universal trainer for earnings call prediction models.

    Handles:
    - Kalshi contract discovery and validation
    - Transcript fetching (real or mock)
    - Feature extraction
    - Model training with walk-forward backtesting
    - Results reporting
    """

    def __init__(self, config: TrainingConfig):
        self.config = config
        self.ticker = config.ticker.upper()
        self.output_dir = Path(config.output_dir) / self.ticker
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Will be populated during training
        self.contracts: List[EarningsContractWord] = []
        self.contract_words: List[str] = []
        self.kalshi_available: bool = False
        self.market_prices: Optional[pd.DataFrame] = None

    def log(self, message: str, level: str = "info"):
        """Log a message if verbose mode is enabled."""
        if self.config.verbose:
            prefix = {"info": "", "success": "[OK] ", "warning": "[!] ", "error": "[X] "}
            print(f"{prefix.get(level, '')}{message}")

    async def discover_kalshi_contracts(self) -> List[EarningsContractWord]:
        """
        Discover available Kalshi contracts for this ticker.

        Returns list of EarningsContractWord if contracts exist,
        empty list if no contracts found.
        """
        self.log(f"\nDiscovering Kalshi contracts for {self.ticker}...")

        try:
            client = get_kalshi_client()
            series_ticker = f"KXEARNINGSMENTION{self.ticker}"

            if isinstance(client, KalshiSdkAdapter):
                markets = await client.get_markets_async(series_ticker=series_ticker)
            else:
                markets = client.get_markets(series_ticker=series_ticker)

            if not markets:
                self.log(f"No Kalshi contracts found for series {series_ticker}", "warning")
                return []

            self.log(f"Found {len(markets)} markets for {series_ticker}", "success")
            self.kalshi_available = True

            # Parse contracts
            contracts = {}
            market_prices_data = {}

            for market in markets:
                ticker = market.get("ticker", "")
                status = market.get("status", "")
                custom_strike = market.get("custom_strike", {})
                word = custom_strike.get("Word", None)

                if not word:
                    continue

                # Normalize word
                word_lower = word.lower().strip()

                if word_lower not in contracts:
                    contracts[word_lower] = EarningsContractWord(
                        word=word_lower,
                        ticker=self.ticker,
                        market_ticker=ticker,
                        market_title=market.get("title", ""),
                        threshold=1,  # Binary contracts
                        markets=[],
                    )

                contracts[word_lower].markets.append(market)

                # Extract market prices for backtesting
                last_price = market.get("last_price", 50) / 100  # Convert cents to probability
                if word_lower not in market_prices_data:
                    market_prices_data[word_lower] = []
                market_prices_data[word_lower].append({
                    "ticker": ticker,
                    "status": status,
                    "price": last_price,
                    "expiration": market.get("expiration_time", ""),
                })

            self.contracts = list(contracts.values())
            self.contract_words = [c.word for c in self.contracts]

            self.log(f"\nWords being tracked for {self.ticker}:")
            for word in self.contract_words:
                num_markets = len(contracts[word].markets)
                self.log(f"  - {word} ({num_markets} contracts)")

            return self.contracts

        except Exception as e:
            self.log(f"Error discovering Kalshi contracts: {e}", "error")
            return []

    def generate_mock_contracts(self) -> List[EarningsContractWord]:
        """Generate mock contracts for testing when Kalshi contracts unavailable."""
        self.log("\nGenerating mock contracts for testing...")

        # Common earnings call keywords
        mock_words = ["ai", "revenue", "growth", "margin", "innovation", "cloud", "efficiency"]

        self.contracts = []
        for word in mock_words:
            contract = EarningsContractWord(
                word=word,
                ticker=self.ticker,
                market_ticker=f"MOCK-{self.ticker}-{word.upper()}",
                market_title=f"Will {self.ticker} mention '{word}'?",
                threshold=1,
                markets=[],
            )
            self.contracts.append(contract)

        self.contract_words = [c.word for c in self.contracts]
        self.log(f"Created {len(self.contracts)} mock contracts: {', '.join(self.contract_words)}")

        return self.contracts

    def generate_mock_transcripts(self, call_dates: List[str]) -> Path:
        """Generate mock earnings call transcripts for testing."""
        segments_dir = self.output_dir / "segments"
        segments_dir.mkdir(parents=True, exist_ok=True)

        self.log(f"\nGenerating {len(call_dates)} mock transcripts...")

        # Templates for generating realistic text
        ceo_templates = [
            "We're extremely pleased with our {word} performance this quarter.",
            "Our focus on {word} continues to deliver strong results.",
            "Looking ahead, {word} remains a key strategic priority.",
            "The team has executed exceptionally well on {word}.",
            "We're seeing tremendous momentum in {word}.",
        ]

        cfo_templates = [
            "From a financial perspective, {word} contributed significantly.",
            "Our {word} metrics show solid improvement quarter over quarter.",
            "We're seeing positive trends in {word} across all segments.",
            "This contributed meaningfully to our overall margins.",
        ]

        analyst_templates = [
            "Can you provide more color on {word}?",
            "What's your outlook for {word} going forward?",
            "How does {word} factor into your guidance?",
        ]

        np.random.seed(42)  # Reproducible results

        for call_date in call_dates:
            segments = []
            n_segments = np.random.randint(25, 40)

            for seg_idx in range(n_segments):
                role_choice = np.random.choice(["ceo", "cfo", "analyst"], p=[0.4, 0.35, 0.25])

                if role_choice == "ceo":
                    speaker, role = "CEO", "ceo"
                    templates = ceo_templates
                elif role_choice == "cfo":
                    speaker, role = "CFO", "cfo"
                    templates = cfo_templates
                else:
                    speaker, role = "Analyst", "analyst"
                    templates = analyst_templates

                # Generate sentences with varying word mentions
                sentences = []
                n_sentences = np.random.randint(2, 5)

                for _ in range(n_sentences):
                    word = np.random.choice(self.contract_words)
                    template = np.random.choice(templates)

                    # Probability of mentioning the word varies by word
                    mention_prob = 0.3 + np.random.uniform(-0.1, 0.2)

                    if np.random.random() < mention_prob:
                        sentence = template.format(word=word)
                    else:
                        sentence = template.format(word="our operations")

                    sentences.append(sentence)

                segment = {
                    "speaker": speaker,
                    "role": role,
                    "text": " ".join(sentences),
                    "segment_idx": seg_idx,
                }
                segments.append(segment)

            # Save transcript
            output_file = segments_dir / f"{self.ticker}_{call_date}.jsonl"
            with open(output_file, "w") as f:
                for seg in segments:
                    f.write(json.dumps(seg) + "\n")

        self.log(f"Generated {len(call_dates)} transcripts in {segments_dir}", "success")
        return segments_dir

    def load_or_generate_transcripts(self, call_dates: List[str]) -> Path:
        """Load existing transcripts or generate mock ones."""
        segments_dir = self.output_dir / "segments"

        # Check for existing transcripts
        if segments_dir.exists():
            existing = list(segments_dir.glob(f"{self.ticker}_*.jsonl"))
            if len(existing) >= len(call_dates):
                self.log(f"Using {len(existing)} existing transcripts from {segments_dir}")
                return segments_dir

        # Generate mock transcripts
        return self.generate_mock_transcripts(call_dates)

    def analyze_transcripts(
        self,
        segments_dir: Path,
        call_dates: List[str],
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Analyze transcripts to extract features and outcomes.

        Returns:
            features_df: Word counts per call
            outcomes_df: Binary outcomes (1 if word mentioned, 0 otherwise)
        """
        self.log("\nAnalyzing transcripts for word mentions...")

        features_data = []
        outcomes_data = []

        for call_date in call_dates:
            segment_file = segments_dir / f"{self.ticker}_{call_date}.jsonl"

            if not segment_file.exists():
                self.log(f"Warning: Missing transcript for {call_date}", "warning")
                continue

            # Load segments
            segments = []
            with open(segment_file, "r") as f:
                for line in f:
                    segments.append(json.loads(line))

            # Filter to executives only (CEO, CFO)
            exec_segments = [
                seg for seg in segments
                if seg.get("role") in ("ceo", "cfo", "executive")
            ]
            combined_text = " ".join(seg["text"] for seg in exec_segments)

            # Count each contract word
            feature_row = {}
            outcome_row = {}

            for word in self.contract_words:
                # Handle multi-word phrases (e.g., "VR / Virtual Reality")
                if "/" in word:
                    variants = [v.strip().lower() for v in word.split("/")]
                else:
                    variants = [word.lower()]

                count = 0
                for variant in variants:
                    pattern = r"\b" + re.escape(variant) + r"\b"
                    count += len(re.findall(pattern, combined_text, re.IGNORECASE))

                feature_row[word] = count
                outcome_row[word] = 1 if count >= 1 else 0  # Binary: mentioned at all

            features_data.append(feature_row)
            outcomes_data.append(outcome_row)

        features_df = pd.DataFrame(features_data, index=call_dates[:len(features_data)])
        outcomes_df = pd.DataFrame(outcomes_data, index=call_dates[:len(outcomes_data)])

        self.log(f"\nFeature matrix: {features_df.shape[0]} calls x {features_df.shape[1]} words")
        self.log(f"Outcome matrix: {outcomes_df.shape[0]} calls x {outcomes_df.shape[1]} words")

        # Print mention rates
        self.log("\nWord mention rates (executives only):")
        for word in self.contract_words:
            if word in outcomes_df.columns:
                rate = outcomes_df[word].mean()
                avg_count = features_df[word].mean() if word in features_df.columns else 0
                self.log(f"  {word}: {rate:.1%} of calls (avg {avg_count:.1f} mentions)")

        return features_df, outcomes_df

    def extract_market_features(
        self,
        call_dates: List[str],
        outcomes_df: pd.DataFrame,
    ) -> Optional[pd.DataFrame]:
        """Extract stock and earnings features for each call date."""
        if not self.config.use_market_features:
            return None

        self.log("\nExtracting market features (stock prices, earnings data)...")

        try:
            extractor = MarketFeatureExtractor(
                cache_dir=self.output_dir / "market_cache",
                use_cached=True,
            )

            records = []
            for call_date in call_dates:
                # Extract stock features
                stock_features = extractor.calculate_stock_features(self.ticker, call_date)
                earnings_features = extractor.calculate_earnings_features(self.ticker, call_date)

                # Extract prior mention features for each word
                for word in self.contract_words:
                    mention_features = extractor.calculate_prior_mention_features(
                        word, call_date, outcomes_df
                    )

                    record = {
                        "call_date": call_date,
                        "word": word,
                        **stock_features,
                        **earnings_features,
                        **mention_features,
                    }
                    records.append(record)

            market_features_df = pd.DataFrame(records)

            # Log summary
            if not market_features_df.empty:
                self.log(f"Extracted {len(market_features_df)} feature records")

                # Show feature availability
                for col in ["stock_return_30d", "eps_surprise_last", "mentioned_last_call"]:
                    if col in market_features_df.columns:
                        non_null = market_features_df[col].notna().sum()
                        self.log(f"  {col}: {non_null}/{len(market_features_df)} available")

            return market_features_df

        except Exception as e:
            self.log(f"Warning: Could not extract market features: {e}", "warning")
            return None

    def run_backtest(
        self,
        features_df: pd.DataFrame,
        outcomes_df: pd.DataFrame,
        market_features_df: Optional[pd.DataFrame] = None,
    ) -> BacktestResult:
        """Run walk-forward backtest with configured parameters."""
        self.log("\n" + "=" * 60)
        self.log("RUNNING BACKTEST")
        self.log("=" * 60)

        # Select model class based on configuration
        if self.config.use_market_features and market_features_df is not None:
            model_class = FeatureAwareEarningsModel
            model_name = "FeatureAwareEarningsModel"
            model_params = {
                "alpha_prior": self.config.alpha_prior,
                "beta_prior": self.config.beta_prior,
                "half_life": self.config.half_life,
                "market_weight": self.config.market_weight,
                "use_features": True,
            }
        else:
            model_class = BetaBinomialEarningsModel
            model_name = "BetaBinomialEarningsModel"
            model_params = {
                "alpha_prior": self.config.alpha_prior,
                "beta_prior": self.config.beta_prior,
                "half_life": self.config.half_life,
            }

        self.log(f"\nModel: {model_name}")
        self.log(f"Parameters:")
        self.log(f"  alpha_prior: {self.config.alpha_prior}")
        self.log(f"  beta_prior: {self.config.beta_prior}")
        self.log(f"  half_life: {self.config.half_life}")
        if self.config.use_market_features:
            self.log(f"  market_weight: {self.config.market_weight}")
            self.log(f"  market_features: ENABLED")

        if self.config.use_enhanced_backtester:
            self.log(f"\nUsing ENHANCED backtester with Kelly criterion")
            self.log(f"Enhanced parameters:")
            self.log(f"  kelly_fraction: {self.config.kelly_fraction} (of full Kelly)")
            self.log(f"  max_position_pct: {self.config.max_position_pct:.1%}")
            self.log(f"  confidence_scaling: {self.config.confidence_scaling}")
            self.log(f"  correlation_limit: {self.config.correlation_limit:.1%}")
            self.log(f"  max_drawdown_limit: {self.config.max_drawdown_limit:.1%}")
            self.log(f"  yes_edge_threshold: {self.config.yes_edge_threshold}")
            self.log(f"  no_edge_threshold: {self.config.no_edge_threshold}")

            backtester = EnhancedEarningsBacktester(
                features=features_df,
                outcomes=outcomes_df,
                model_class=model_class,
                model_params=model_params,
                kelly_fraction=self.config.kelly_fraction,
                max_position_pct=self.config.max_position_pct,
                confidence_scaling=self.config.confidence_scaling,
                correlation_limit=self.config.correlation_limit,
                max_drawdown_limit=self.config.max_drawdown_limit,
                yes_edge_threshold=self.config.yes_edge_threshold,
                no_edge_threshold=self.config.no_edge_threshold,
                fee_rate=self.config.fee_rate,
                min_train_window=self.config.min_train_window,
            )
        else:
            self.log(f"\nTrading parameters:")
            self.log(f"  edge_threshold: {self.config.edge_threshold}")
            self.log(f"  yes_edge_threshold: {self.config.yes_edge_threshold}")
            self.log(f"  no_edge_threshold: {self.config.no_edge_threshold}")
            self.log(f"  position_size_pct: {self.config.position_size_pct}")
            self.log(f"  initial_capital: ${self.config.initial_capital:,.2f}")

            backtester = EarningsKalshiBacktester(
                features=features_df,
                outcomes=outcomes_df,
                model_class=model_class,
                model_params=model_params,
                edge_threshold=self.config.edge_threshold,
                yes_edge_threshold=self.config.yes_edge_threshold,
                no_edge_threshold=self.config.no_edge_threshold,
                position_size_pct=self.config.position_size_pct,
                fee_rate=self.config.fee_rate,
                min_train_window=self.config.min_train_window,
            )

        # Run backtest
        result = backtester.run(
            ticker=self.ticker,
            initial_capital=self.config.initial_capital,
            market_prices=self.market_prices,
        )

        return result

    def print_results(self, result: BacktestResult):
        """Print formatted backtest results."""
        metrics = result.metrics

        print("\n" + "=" * 60)
        print(f"BACKTEST RESULTS: {self.ticker}")
        if self.config.use_enhanced_backtester:
            print("(Enhanced Backtester with Kelly Criterion)")
        print("=" * 60)

        print(f"\nPrediction Performance:")
        print(f"  Total Predictions: {metrics.total_predictions}")
        print(f"  Correct Predictions: {metrics.correct_predictions}")
        print(f"  Accuracy: {metrics.accuracy:.1%}")
        print(f"  Brier Score: {metrics.brier_score:.4f}")

        print(f"\nTrading Performance:")
        print(f"  Total Trades: {metrics.total_trades}")
        print(f"  Winning Trades: {metrics.winning_trades}")
        print(f"  Win Rate: {metrics.win_rate:.1%}")
        print(f"  Total P&L: ${metrics.total_pnl:,.2f}")
        print(f"  Avg P&L/Trade: ${metrics.avg_pnl_per_trade:,.2f}")
        print(f"  ROI: {metrics.roi:.1%}")
        print(f"  Sharpe Ratio: {metrics.sharpe_ratio:.2f}")

        # Enhanced metrics (if available)
        if hasattr(metrics, "max_drawdown"):
            print(f"\nRisk Metrics (Enhanced):")
            print(f"  Max Drawdown: {metrics.max_drawdown:.1%}")
            print(f"  Calmar Ratio: {metrics.calmar_ratio:.2f}")
            print(f"  Avg Kelly Fraction: {metrics.avg_kelly_fraction:.2%}")
            print(f"  Calibration Error: {metrics.calibration_error:.3f}")

        # Show sample trades
        if result.trades:
            print(f"\nRecent Trades (last 5):")
            for trade in result.trades[-5:]:
                outcome_str = "WIN" if trade.pnl > 0 else "LOSS"
                # Check if enhanced trade
                kelly_info = ""
                if hasattr(trade, "kelly_fraction"):
                    kelly_info = f" | Kelly: {trade.kelly_fraction:.1%}"
                print(f"  {trade.call_date} | {trade.contract} | "
                      f"{trade.side} @ {trade.entry_price:.2f} | "
                      f"Edge: {trade.edge:+.2f}{kelly_info} | P&L: ${trade.pnl:+.2f} ({outcome_str})")

    def save_results(self, training_result: TrainingResult):
        """Save all results to output directory."""
        timestamp = training_result.run_timestamp

        # Save main results
        results_file = self.output_dir / f"training_results_{timestamp}.json"
        with open(results_file, "w") as f:
            json.dump(training_result.to_dict(), f, indent=2, default=str)

        # Save backtest results using the appropriate save function
        backtest_dir = self.output_dir / "backtest"
        if self.config.use_enhanced_backtester:
            save_enhanced_backtest_result(training_result.backtest_result, backtest_dir)
        else:
            save_earnings_backtest_result(training_result.backtest_result, backtest_dir)

        # Save configuration
        config_file = self.output_dir / "config.json"
        with open(config_file, "w") as f:
            json.dump(asdict(self.config), f, indent=2, default=str)

        self.log(f"\nResults saved to {self.output_dir}", "success")

    async def train(self) -> TrainingResult:
        """
        Run the complete training pipeline.

        Steps:
        1. Discover/validate Kalshi contracts
        2. Generate call dates
        3. Load or generate transcripts
        4. Analyze transcripts for features/outcomes
        5. Run backtest
        6. Save and return results
        """
        print("=" * 60)
        print(f"EARNINGS MODEL TRAINING: {self.ticker}")
        print("=" * 60)
        print(f"\nTimestamp: {datetime.now().isoformat()}")
        print(f"Configuration:")
        print(f"  Ticker: {self.ticker}")
        print(f"  Quarters: {self.config.num_quarters}")
        print(f"  Use Real Transcripts: {self.config.use_real_transcripts}")
        print(f"  Output Directory: {self.output_dir}")

        # Step 1: Discover contracts
        if self.config.use_real_contracts:
            contracts = await self.discover_kalshi_contracts()
            if not contracts:
                self.log("No Kalshi contracts found, using mock contracts", "warning")
                contracts = self.generate_mock_contracts()
        else:
            contracts = self.generate_mock_contracts()

        # Step 2: Generate call dates
        base_date = datetime.now() - timedelta(days=self.config.num_quarters * 90)
        call_dates = [
            (base_date + timedelta(days=90 * i)).strftime("%Y-%m-%d")
            for i in range(self.config.num_quarters)
        ]
        self.log(f"\nCall dates: {call_dates[0]} to {call_dates[-1]}")

        # Step 3: Load or generate transcripts
        if self.config.use_real_transcripts:
            # TODO: Implement real transcript fetching
            self.log("Real transcript fetching not yet implemented, using mock data", "warning")

        segments_dir = self.load_or_generate_transcripts(call_dates)

        # Step 4: Analyze transcripts
        features_df, outcomes_df = self.analyze_transcripts(segments_dir, call_dates)

        # Save features and outcomes
        features_df.to_csv(self.output_dir / "features.csv")
        outcomes_df.to_csv(self.output_dir / "outcomes.csv")

        # Step 4b: Extract market features (if enabled)
        market_features_df = self.extract_market_features(call_dates, outcomes_df)
        if market_features_df is not None:
            market_features_df.to_csv(self.output_dir / "market_features.csv")

        # Step 5: Run backtest
        backtest_result = self.run_backtest(features_df, outcomes_df, market_features_df)

        # Print results
        self.print_results(backtest_result)

        # Create training result
        training_result = TrainingResult(
            ticker=self.ticker,
            config=self.config,
            contracts_found=len(contracts),
            words_tracked=self.contract_words,
            backtest_result=backtest_result,
            run_timestamp=datetime.now().strftime("%Y%m%d_%H%M%S"),
            kalshi_contracts_available=self.kalshi_available,
        )

        # Step 6: Save results
        self.save_results(training_result)

        print("\n" + "=" * 60)
        print("TRAINING COMPLETE")
        print("=" * 60)
        print(f"\nSummary:")
        print(f"  Ticker: {self.ticker}")
        print(f"  Kalshi Contracts Available: {'Yes' if self.kalshi_available else 'No (using mock)'}")
        print(f"  Words Tracked: {len(self.contract_words)}")
        print(f"  Calls Analyzed: {len(call_dates)}")
        print(f"  Accuracy: {backtest_result.metrics.accuracy:.1%}")
        print(f"  ROI: {backtest_result.metrics.roi:.1%}")
        print(f"\nResults saved to: {self.output_dir}")

        return training_result


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train earnings call prediction model for any ticker",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python scripts/train_earnings_model.py AAPL
    python scripts/train_earnings_model.py TSLA --num-quarters 16
    python scripts/train_earnings_model.py META --half-life 6.0 --edge-threshold 0.15
    python scripts/train_earnings_model.py NVDA --no-real-contracts  # Use mock contracts
        """,
    )

    parser.add_argument(
        "ticker",
        type=str,
        help="Stock ticker symbol (e.g., AAPL, TSLA, META)",
    )

    parser.add_argument(
        "--num-quarters", "-n",
        type=int,
        default=12,
        help="Number of quarters for training (default: 12)",
    )

    parser.add_argument(
        "--use-real-transcripts",
        action="store_true",
        help="Attempt to fetch real transcripts (not yet implemented)",
    )

    parser.add_argument(
        "--no-real-contracts",
        action="store_true",
        help="Use mock contracts instead of checking Kalshi",
    )

    # Model parameters
    parser.add_argument(
        "--alpha-prior",
        type=float,
        default=1.0,
        help="Beta distribution alpha prior (default: 1.0)",
    )

    parser.add_argument(
        "--beta-prior",
        type=float,
        default=1.0,
        help="Beta distribution beta prior (default: 1.0)",
    )

    parser.add_argument(
        "--half-life",
        type=float,
        default=8.0,
        help="Recency weighting half-life in quarters (default: 8.0)",
    )

    # Trading parameters
    parser.add_argument(
        "--edge-threshold",
        type=float,
        default=0.12,
        help="Minimum edge to trade (default: 0.12)",
    )

    parser.add_argument(
        "--yes-edge-threshold",
        type=float,
        default=0.22,
        help="Edge threshold for YES trades (default: 0.22)",
    )

    parser.add_argument(
        "--no-edge-threshold",
        type=float,
        default=0.08,
        help="Edge threshold for NO trades (default: 0.08)",
    )

    parser.add_argument(
        "--position-size",
        type=float,
        default=0.03,
        help="Position size as fraction of capital (default: 0.03)",
    )

    parser.add_argument(
        "--initial-capital",
        type=float,
        default=10000.0,
        help="Starting capital (default: 10000.0)",
    )

    parser.add_argument(
        "--min-train-window",
        type=int,
        default=4,
        help="Minimum quarters for training before predicting (default: 4)",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/training",
        help="Output directory for results (default: data/training)",
    )

    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress verbose output",
    )

    # Enhanced backtester options
    parser.add_argument(
        "--enhanced",
        action="store_true",
        help="Use enhanced backtester with Kelly criterion sizing",
    )

    parser.add_argument(
        "--kelly-fraction",
        type=float,
        default=0.25,
        help="Fraction of Kelly criterion to bet (default: 0.25 = quarter Kelly)",
    )

    parser.add_argument(
        "--max-position",
        type=float,
        default=0.10,
        help="Maximum position size as fraction of capital (default: 0.10)",
    )

    parser.add_argument(
        "--no-confidence-scaling",
        action="store_true",
        help="Disable confidence-based position scaling",
    )

    parser.add_argument(
        "--correlation-limit",
        type=float,
        default=0.25,
        help="Maximum exposure to correlated contracts (default: 0.25)",
    )

    parser.add_argument(
        "--max-drawdown",
        type=float,
        default=0.20,
        help="Stop trading if drawdown exceeds this (default: 0.20)",
    )

    # Market features options
    parser.add_argument(
        "--market-features",
        action="store_true",
        help="Enable stock/earnings external features (requires yfinance)",
    )

    parser.add_argument(
        "--market-weight",
        type=float,
        default=0.3,
        help="Weight for market price signal (0-1, default: 0.3)",
    )

    return parser.parse_args()


async def main():
    """Main entry point."""
    args = parse_args()

    # Build configuration
    config = TrainingConfig(
        ticker=args.ticker.upper(),
        num_quarters=args.num_quarters,
        use_real_transcripts=args.use_real_transcripts,
        use_real_contracts=not args.no_real_contracts,
        use_enhanced_backtester=args.enhanced,
        use_market_features=args.market_features,
        alpha_prior=args.alpha_prior,
        beta_prior=args.beta_prior,
        half_life=args.half_life,
        market_weight=args.market_weight,
        edge_threshold=args.edge_threshold,
        yes_edge_threshold=args.yes_edge_threshold,
        no_edge_threshold=args.no_edge_threshold,
        position_size_pct=args.position_size,
        initial_capital=args.initial_capital,
        min_train_window=args.min_train_window,
        kelly_fraction=args.kelly_fraction,
        max_position_pct=args.max_position,
        confidence_scaling=not args.no_confidence_scaling,
        correlation_limit=args.correlation_limit,
        max_drawdown_limit=args.max_drawdown,
        output_dir=Path(args.output_dir),
        verbose=not args.quiet,
    )

    # Create trainer and run
    trainer = EarningsModelTrainer(config)
    result = await trainer.train()

    return result


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
