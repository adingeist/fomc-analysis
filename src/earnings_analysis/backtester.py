"""
Backtester for earnings call trading strategies.

Similar to FOMC backtester but adapted for earnings calls and stock price movements.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Dict

import numpy as np
import pandas as pd

from .models.base import EarningsModel


@dataclass
class EarningsPrediction:
    """A prediction for an earnings call."""
    ticker: str
    call_date: str
    predicted_probability: float
    confidence_lower: float
    confidence_upper: float
    actual_outcome: Optional[int] = None  # 1 if price up, 0 if down
    actual_return: Optional[float] = None  # Actual price return
    correct: Optional[bool] = None


@dataclass
class EarningsTrade:
    """Record of a trade execution."""
    ticker: str
    call_date: str
    side: str  # "LONG" or "SHORT"
    position_size: float
    entry_price: float
    exit_price: Optional[float] = None
    predicted_probability: float
    edge: float
    actual_return: float = 0.0
    pnl: float = 0.0
    roi: float = 0.0


@dataclass
class BacktestResult:
    """Complete backtest results."""
    predictions: List[EarningsPrediction]
    trades: List[EarningsTrade]
    metrics: Dict[str, float]
    metadata: Dict[str, any]


class EarningsBacktester:
    """
    Backtest earnings call trading strategies.

    Parameters
    ----------
    features : pd.DataFrame
        Features dataframe (index = call_date)
    outcomes : pd.DataFrame
        Price outcomes dataframe (index = call_date)
    model_class : type
        Model class to use
    model_params : dict
        Model parameters
    edge_threshold : float
        Minimum edge to trade (default: 0.1 = 10%)
    position_size_pct : float
        Percentage of capital per trade (default: 0.05 = 5%)
    transaction_cost : float
        Transaction cost per trade (default: 0.001 = 0.1%)
    min_train_window : int
        Minimum number of historical calls for training
    test_start_date : Optional[str]
        Start date for out-of-sample testing
    """

    def __init__(
        self,
        features: pd.DataFrame,
        outcomes: pd.DataFrame,
        model_class: type,
        model_params: dict = None,
        edge_threshold: float = 0.1,
        position_size_pct: float = 0.05,
        transaction_cost: float = 0.001,
        min_train_window: int = 4,
        test_start_date: Optional[str] = None,
    ):
        self.features = features.sort_index()
        self.outcomes = outcomes.sort_index()
        self.model_class = model_class
        self.model_params = model_params or {}
        self.edge_threshold = edge_threshold
        self.position_size_pct = position_size_pct
        self.transaction_cost = transaction_cost
        self.min_train_window = min_train_window
        self.test_start_date = pd.to_datetime(test_start_date) if test_start_date else None

        # Merge features and outcomes
        self.data = self._merge_data()

    def _merge_data(self) -> pd.DataFrame:
        """Merge features and outcomes on call_date."""
        # Ensure outcomes has the outcome column
        if "direction_1d" in self.outcomes.columns:
            outcome_col = "direction_1d"
        elif "direction_5d" in self.outcomes.columns:
            outcome_col = "direction_5d"
        else:
            raise ValueError("Outcomes must have direction_1d or direction_5d column")

        # Merge on index (call_date)
        merged = self.features.join(
            self.outcomes[[outcome_col, "return_1d"]],
            how="inner"
        )

        merged = merged.rename(columns={outcome_col: "outcome"})

        return merged

    def run(self, initial_capital: float = 10000.0) -> BacktestResult:
        """
        Run walk-forward backtest.

        Returns
        -------
        BacktestResult
            Complete backtest results
        """
        capital = initial_capital
        predictions = []
        trades = []

        call_dates = sorted(self.data.index)

        # Walk forward through earnings calls
        for i, current_date in enumerate(call_dates):
            # Skip if not enough historical data
            if i < self.min_train_window:
                continue

            # Skip if before test start date
            if self.test_start_date and pd.to_datetime(current_date) < self.test_start_date:
                continue

            # Train on all previous data
            train_data = self.data.iloc[:i]

            # Get features and outcomes for training
            train_features = train_data.drop(columns=["outcome", "return_1d"], errors="ignore")
            train_outcomes = train_data["outcome"]

            # Fit model
            model = self.model_class(**self.model_params)
            model.fit(train_features, train_outcomes)

            # Make prediction
            current_features = self.data.loc[[current_date]].drop(
                columns=["outcome", "return_1d"],
                errors="ignore"
            )
            pred = model.predict(current_features)

            predicted_prob = float(pred.iloc[0]["probability"])
            lower_bound = float(pred.iloc[0]["lower_bound"])
            upper_bound = float(pred.iloc[0]["upper_bound"])

            # Get actual outcome
            actual_outcome = int(self.data.loc[current_date, "outcome"])
            actual_return = float(self.data.loc[current_date, "return_1d"])

            # Determine if prediction was correct
            predicted_direction = 1 if predicted_prob > 0.5 else 0
            correct = (predicted_direction == actual_outcome)

            # Create prediction record
            ticker = self.data.loc[current_date, "ticker"] if "ticker" in self.data.columns else "UNKNOWN"

            prediction = EarningsPrediction(
                ticker=ticker,
                call_date=str(current_date),
                predicted_probability=predicted_prob,
                confidence_lower=lower_bound,
                confidence_upper=upper_bound,
                actual_outcome=actual_outcome,
                actual_return=actual_return,
                correct=correct,
            )
            predictions.append(prediction)

            # Trading logic
            # Edge = abs(predicted_prob - 0.5) * 2  # Scale to 0-1
            edge = abs(predicted_prob - 0.5)

            if edge >= self.edge_threshold:
                # Determine side
                if predicted_prob > 0.5:
                    side = "LONG"
                    direction_multiplier = 1
                else:
                    side = "SHORT"
                    direction_multiplier = -1

                # Position sizing
                position_size = capital * self.position_size_pct

                # Simulate trade
                pnl = position_size * actual_return * direction_multiplier
                pnl -= position_size * self.transaction_cost  # Transaction costs

                capital += pnl
                roi = pnl / position_size if position_size > 0 else 0

                trade = EarningsTrade(
                    ticker=ticker,
                    call_date=str(current_date),
                    side=side,
                    position_size=position_size,
                    entry_price=1.0,  # Normalized
                    predicted_probability=predicted_prob,
                    edge=edge,
                    actual_return=actual_return,
                    pnl=pnl,
                    roi=roi,
                )
                trades.append(trade)

        # Calculate metrics
        metrics = self._calculate_metrics(trades, initial_capital, capital)

        # Create result
        result = BacktestResult(
            predictions=predictions,
            trades=trades,
            metrics=metrics,
            metadata={
                "model_class": self.model_class.__name__,
                "model_params": self.model_params,
                "initial_capital": initial_capital,
                "final_capital": capital,
                "edge_threshold": self.edge_threshold,
                "position_size_pct": self.position_size_pct,
                "transaction_cost": self.transaction_cost,
            },
        )

        return result

    def _calculate_metrics(
        self,
        trades: List[EarningsTrade],
        initial_capital: float,
        final_capital: float,
    ) -> Dict[str, float]:
        """Calculate backtest metrics."""
        if not trades:
            return {
                "total_trades": 0,
                "win_rate": 0.0,
                "total_pnl": 0.0,
                "avg_pnl": 0.0,
                "roi": 0.0,
                "sharpe": 0.0,
            }

        total_trades = len(trades)
        winning_trades = len([t for t in trades if t.pnl > 0])
        win_rate = winning_trades / total_trades

        total_pnl = sum(t.pnl for t in trades)
        avg_pnl = total_pnl / total_trades

        roi = (final_capital - initial_capital) / initial_capital

        # Sharpe ratio
        returns = [t.roi for t in trades]
        sharpe = np.mean(returns) / np.std(returns) * np.sqrt(len(returns)) if np.std(returns) > 0 else 0

        return {
            "total_trades": total_trades,
            "win_rate": float(win_rate),
            "total_pnl": float(total_pnl),
            "avg_pnl": float(avg_pnl),
            "roi": float(roi),
            "sharpe": float(sharpe),
            "final_capital": float(final_capital),
        }


def save_backtest_result(result: BacktestResult, output_dir: Path):
    """Save backtest results to files."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save main results
    data = {
        "predictions": [asdict(p) for p in result.predictions],
        "trades": [asdict(t) for t in result.trades],
        "metrics": result.metrics,
        "metadata": result.metadata,
    }

    (output_dir / "backtest_results.json").write_text(
        json.dumps(data, indent=2, default=str)
    )

    # Save predictions as CSV
    if result.predictions:
        pred_df = pd.DataFrame([asdict(p) for p in result.predictions])
        pred_df.to_csv(output_dir / "predictions.csv", index=False)

    # Save trades as CSV
    if result.trades:
        trade_df = pd.DataFrame([asdict(t) for t in result.trades])
        trade_df.to_csv(output_dir / "trades.csv", index=False)
