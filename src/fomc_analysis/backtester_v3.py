"""
Time-horizon backtester with Kalshi contract outcomes.

This backtester implements a more realistic testing framework:
- Fetches actual Kalshi contract outcomes (100% YES or 0% NO)
- Makes predictions at specific time horizons: 7, 14, 30 days before meetings
- Tracks prediction accuracy for each time horizon
- Simulates realistic trading with Kalshi fees
- Provides comprehensive profitability analysis

This addresses the limitations of backtester_v2.py by:
1. Using actual market outcomes instead of just transcript features
2. Testing predictions at realistic time intervals before meetings
3. Providing better accuracy metrics per time horizon
"""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional, Dict, Tuple

import numpy as np
import pandas as pd


@dataclass
class PredictionSnapshot:
    """A prediction made at a specific time before a meeting."""
    meeting_date: str
    contract: str
    prediction_date: str  # When the prediction was made
    days_before_meeting: int  # 7, 14, or 30
    predicted_probability: float  # Model's probability estimate
    confidence_lower: float  # Lower confidence bound
    confidence_upper: float  # Upper confidence bound
    actual_outcome: Optional[int] = None  # 1 if YES, 0 if NO (from Kalshi)
    market_price: Optional[float] = None  # Kalshi market price at prediction time
    edge: Optional[float] = None  # predicted_probability - market_price
    correct: Optional[bool] = None  # True if prediction was correct


@dataclass
class Trade:
    """Record of a single trade execution."""
    meeting_date: str
    contract: str
    prediction_date: str
    days_before_meeting: int
    side: str  # "YES" or "NO"
    position_size: float  # Dollars invested
    entry_price: float  # Price paid (0-1)
    predicted_probability: float
    edge: float
    actual_outcome: int  # 1 if YES, 0 if NO
    pnl: float
    roi: float  # Return on investment for this trade


@dataclass
class HorizonMetrics:
    """Performance metrics for a specific time horizon."""
    horizon_days: int
    total_predictions: int
    correct_predictions: int
    accuracy: float
    total_trades: int
    winning_trades: int
    win_rate: float
    total_pnl: float
    avg_pnl_per_trade: float
    roi: float
    sharpe_ratio: float
    brier_score: float  # Calibration metric


@dataclass
class BacktestResult:
    """Complete backtest results."""
    predictions: List[PredictionSnapshot]
    trades: List[Trade]
    horizon_metrics: Dict[int, HorizonMetrics]  # Keyed by days_before_meeting
    overall_metrics: Dict[str, float]
    metadata: Dict[str, any]


def fetch_kalshi_contract_outcomes(
    contract_words: List[Dict],
    kalshi_client,
) -> pd.DataFrame:
    """
    Fetch actual Kalshi contract outcomes.

    Parameters
    ----------
    contract_words : List[Dict]
        List of contract definitions with market tickers
    kalshi_client : KalshiClientProtocol
        Kalshi API client

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: meeting_date, contract, outcome (0 or 1), close_price
    """
    outcomes = []

    for contract_def in contract_words:
        contract_name = contract_def.get("word", "")
        threshold = contract_def.get("threshold", 1)

        # Display name for the contract
        if threshold > 1:
            display_name = f"{contract_name} ({threshold}+)"
        else:
            display_name = contract_name

        for market in contract_def.get("markets", []):
            ticker = market.get("ticker")
            status = market.get("status", "")

            # Only process resolved markets
            if status != "resolved":
                continue

            # Get the meeting date (expiration/close date)
            meeting_date_str = market.get("expiration_date") or market.get("close_date")
            if not meeting_date_str:
                continue

            try:
                meeting_date = datetime.fromisoformat(meeting_date_str).date()
            except ValueError:
                continue

            # Get the outcome
            result = market.get("result", "")
            if result == "yes":
                outcome = 1
            elif result == "no":
                outcome = 0
            else:
                continue  # Skip unresolved or invalid outcomes

            # Get the final close price if available
            close_price = market.get("close_price")
            if close_price is not None:
                close_price = float(close_price)
                # Normalize to 0-1 if it's in cents
                if close_price > 1:
                    close_price = close_price / 100.0

            outcomes.append({
                "meeting_date": meeting_date,
                "contract": display_name,
                "ticker": ticker,
                "outcome": outcome,
                "close_price": close_price,
            })

    if not outcomes:
        return pd.DataFrame(columns=["meeting_date", "contract", "outcome", "close_price"])

    df = pd.DataFrame(outcomes)
    df["meeting_date"] = pd.to_datetime(df["meeting_date"])
    return df


def fetch_historical_prices_at_horizons(
    tickers: List[str],
    meeting_dates: List[datetime],
    horizons: List[int],  # [7, 14, 30]
    kalshi_client,
) -> pd.DataFrame:
    """
    Fetch historical Kalshi prices at specific days before meetings.

    Parameters
    ----------
    tickers : List[str]
        List of Kalshi market tickers
    meeting_dates : List[datetime]
        List of meeting dates
    horizons : List[int]
        Days before meeting to fetch prices (e.g., [7, 14, 30])
    kalshi_client : KalshiClientProtocol
        Kalshi API client

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: ticker, meeting_date, days_before, price
    """
    price_records = []

    for ticker in tickers:
        try:
            # Fetch full price history for this ticker
            history_df = kalshi_client.get_market_history(ticker)

            if history_df.empty or ticker not in history_df.columns:
                continue

            price_series = history_df[ticker]

            # Normalize prices to 0-1 if needed
            max_price = price_series.max()
            if pd.notna(max_price) and max_price > 1:
                price_series = price_series / 100.0

            # For each meeting date, get prices at each horizon
            for meeting_date in meeting_dates:
                for days_before in horizons:
                    target_date = meeting_date - timedelta(days=days_before)

                    # Find the closest available price on or before target_date
                    available_dates = price_series.index[price_series.index <= pd.Timestamp(target_date)]

                    if len(available_dates) == 0:
                        continue

                    closest_date = available_dates[-1]
                    price = float(price_series.loc[closest_date])

                    price_records.append({
                        "ticker": ticker,
                        "meeting_date": meeting_date,
                        "days_before": days_before,
                        "prediction_date": closest_date.date(),
                        "price": price,
                    })

        except Exception as e:
            print(f"Warning: Could not fetch history for {ticker}: {e}")
            continue

    if not price_records:
        return pd.DataFrame(columns=["ticker", "meeting_date", "days_before", "prediction_date", "price"])

    return pd.DataFrame(price_records)


class TimeHorizonBacktester:
    """
    Backtester that makes predictions at specific time horizons before meetings.

    Parameters
    ----------
    outcomes : pd.DataFrame
        Actual contract outcomes (from Kalshi)
    historical_prices : pd.DataFrame
        Historical market prices at different horizons
    horizons : List[int], default=[7, 14, 30]
        Days before meeting to make predictions
    edge_threshold : float, default=0.10
        Minimum edge required to trade
    position_size_pct : float, default=0.05
        Fraction of capital to risk per trade
    fee_rate : float, default=0.07
        Kalshi fee rate (7% on profits)
    min_train_window : int, default=5
        Minimum number of historical meetings for training
    """

    def __init__(
        self,
        outcomes: pd.DataFrame,
        historical_prices: pd.DataFrame,
        horizons: List[int] = None,
        edge_threshold: float = 0.10,
        position_size_pct: float = 0.05,
        fee_rate: float = 0.07,
        min_train_window: int = 5,
    ):
        self.outcomes = outcomes.sort_values("meeting_date")
        self.historical_prices = historical_prices
        self.horizons = horizons or [7, 14, 30]
        self.edge_threshold = edge_threshold
        self.position_size_pct = position_size_pct
        self.fee_rate = fee_rate
        self.min_train_window = min_train_window

    def run(
        self,
        model_class,
        model_params: Optional[Dict] = None,
        initial_capital: float = 10000.0,
    ) -> BacktestResult:
        """
        Run time-horizon backtest.

        For each meeting:
        1. Train model on all previous meetings
        2. Make predictions at each horizon (7, 14, 30 days before)
        3. Compare predictions with actual outcomes
        4. Simulate trades if edge > threshold
        5. Track accuracy and profitability

        Parameters
        ----------
        model_class : class
            Model class to instantiate (e.g., BetaBinomialModel)
        model_params : Optional[Dict]
            Parameters for model constructor
        initial_capital : float
            Starting capital for trading simulation

        Returns
        -------
        BacktestResult
            Complete backtest results
        """
        model_params = model_params or {}
        capital = initial_capital

        predictions = []
        trades = []

        # Group outcomes by meeting date
        meeting_dates = sorted(self.outcomes["meeting_date"].unique())

        # Build feature matrix from outcomes for model training
        # Pivot outcomes to get contract x meeting matrix
        outcome_matrix = self.outcomes.pivot_table(
            index="meeting_date",
            columns="contract",
            values="outcome",
            aggfunc="first"
        )

        # Walk forward through meetings
        for i, current_meeting in enumerate(meeting_dates):
            # Skip if not enough historical data
            if i < self.min_train_window:
                continue

            # Train on all previous meetings
            train_data = outcome_matrix.loc[outcome_matrix.index < current_meeting]

            # Fit model
            model = model_class(**model_params)
            model.fit(train_data)

            # Get contracts for this meeting
            meeting_outcomes = self.outcomes[self.outcomes["meeting_date"] == current_meeting]

            # Make predictions at each horizon
            for horizon in self.horizons:
                prediction_date = current_meeting - timedelta(days=horizon)

                # Get model predictions
                model_predictions = model.predict(n_future=1)

                # For each contract in this meeting
                for _, outcome_row in meeting_outcomes.iterrows():
                    contract = outcome_row["contract"]
                    actual_outcome = int(outcome_row["outcome"])
                    ticker = outcome_row["ticker"]

                    # Find model prediction for this contract
                    model_pred = model_predictions[model_predictions["contract"] == contract]
                    if model_pred.empty:
                        continue

                    predicted_prob = float(model_pred.iloc[0]["probability"])
                    lower_bound = float(model_pred.iloc[0]["lower_bound"])
                    upper_bound = float(model_pred.iloc[0]["upper_bound"])

                    # Get market price at this horizon
                    price_data = self.historical_prices[
                        (self.historical_prices["ticker"] == ticker) &
                        (self.historical_prices["meeting_date"] == current_meeting) &
                        (self.historical_prices["days_before"] == horizon)
                    ]

                    market_price = None
                    pred_date_actual = None
                    if not price_data.empty:
                        market_price = float(price_data.iloc[0]["price"])
                        pred_date_actual = price_data.iloc[0]["prediction_date"]

                    # Calculate edge
                    edge = predicted_prob - market_price if market_price is not None else None

                    # Determine if prediction was correct (using 50% threshold)
                    predicted_yes = predicted_prob > 0.5
                    correct = (predicted_yes and actual_outcome == 1) or (not predicted_yes and actual_outcome == 0)

                    # Create prediction snapshot
                    snapshot = PredictionSnapshot(
                        meeting_date=current_meeting.strftime("%Y-%m-%d"),
                        contract=contract,
                        prediction_date=pred_date_actual.strftime("%Y-%m-%d") if pred_date_actual else prediction_date.strftime("%Y-%m-%d"),
                        days_before_meeting=horizon,
                        predicted_probability=predicted_prob,
                        confidence_lower=lower_bound,
                        confidence_upper=upper_bound,
                        actual_outcome=actual_outcome,
                        market_price=market_price,
                        edge=edge,
                        correct=correct,
                    )
                    predictions.append(snapshot)

                    # Decide whether to trade
                    if edge is not None and abs(edge) >= self.edge_threshold:
                        # Determine trade direction
                        if edge > 0:
                            side = "YES"
                            entry_price = market_price
                        else:
                            side = "NO"
                            entry_price = 1 - market_price

                        # Position size
                        position_size = capital * self.position_size_pct

                        # Calculate P&L
                        if side == "YES":
                            if actual_outcome == 1:
                                # Won bet
                                gross_pnl = position_size * (1 - entry_price) / entry_price
                                fees = gross_pnl * self.fee_rate
                                pnl = gross_pnl - fees
                            else:
                                # Lost bet
                                pnl = -position_size
                        else:  # NO
                            if actual_outcome == 0:
                                # Won bet
                                gross_pnl = position_size * entry_price / (1 - entry_price)
                                fees = gross_pnl * self.fee_rate
                                pnl = gross_pnl - fees
                            else:
                                # Lost bet
                                pnl = -position_size

                        # Update capital
                        capital += pnl
                        roi = pnl / position_size

                        # Record trade
                        trade = Trade(
                            meeting_date=current_meeting.strftime("%Y-%m-%d"),
                            contract=contract,
                            prediction_date=pred_date_actual.strftime("%Y-%m-%d") if pred_date_actual else prediction_date.strftime("%Y-%m-%d"),
                            days_before_meeting=horizon,
                            side=side,
                            position_size=position_size,
                            entry_price=entry_price,
                            predicted_probability=predicted_prob,
                            edge=edge,
                            actual_outcome=actual_outcome,
                            pnl=pnl,
                            roi=roi,
                        )
                        trades.append(trade)

        # Compute metrics by horizon
        horizon_metrics = self._compute_horizon_metrics(predictions, trades)

        # Compute overall metrics
        overall_metrics = self._compute_overall_metrics(trades, initial_capital, capital)

        return BacktestResult(
            predictions=predictions,
            trades=trades,
            horizon_metrics=horizon_metrics,
            overall_metrics=overall_metrics,
            metadata={
                "model_class": model_class.__name__,
                "model_params": model_params,
                "initial_capital": initial_capital,
                "final_capital": capital,
                "horizons": self.horizons,
                "edge_threshold": self.edge_threshold,
                "position_size_pct": self.position_size_pct,
                "fee_rate": self.fee_rate,
            },
        )

    def _compute_horizon_metrics(
        self,
        predictions: List[PredictionSnapshot],
        trades: List[Trade],
    ) -> Dict[int, HorizonMetrics]:
        """Compute performance metrics for each time horizon."""
        metrics = {}

        for horizon in self.horizons:
            # Filter predictions for this horizon
            horizon_preds = [p for p in predictions if p.days_before_meeting == horizon]
            horizon_trades = [t for t in trades if t.days_before_meeting == horizon]

            if not horizon_preds:
                continue

            # Accuracy
            correct_preds = [p for p in horizon_preds if p.correct]
            accuracy = len(correct_preds) / len(horizon_preds) if horizon_preds else 0

            # Trading metrics
            total_trades = len(horizon_trades)
            winning_trades = len([t for t in horizon_trades if t.pnl > 0])
            win_rate = winning_trades / total_trades if total_trades > 0 else 0

            total_pnl = sum(t.pnl for t in horizon_trades)
            avg_pnl = total_pnl / total_trades if total_trades > 0 else 0

            # ROI
            total_invested = sum(t.position_size for t in horizon_trades)
            roi = total_pnl / total_invested if total_invested > 0 else 0

            # Sharpe ratio
            if horizon_trades:
                returns = [t.roi for t in horizon_trades]
                sharpe = np.mean(returns) / np.std(returns) * np.sqrt(len(returns)) if np.std(returns) > 0 else 0
            else:
                sharpe = 0

            # Brier score (calibration)
            brier_scores = []
            for pred in horizon_preds:
                if pred.actual_outcome is not None:
                    brier_scores.append((pred.predicted_probability - pred.actual_outcome) ** 2)
            brier_score = np.mean(brier_scores) if brier_scores else 0

            metrics[horizon] = HorizonMetrics(
                horizon_days=horizon,
                total_predictions=len(horizon_preds),
                correct_predictions=len(correct_preds),
                accuracy=accuracy,
                total_trades=total_trades,
                winning_trades=winning_trades,
                win_rate=win_rate,
                total_pnl=total_pnl,
                avg_pnl_per_trade=avg_pnl,
                roi=roi,
                sharpe_ratio=sharpe,
                brier_score=brier_score,
            )

        return metrics

    def _compute_overall_metrics(
        self,
        trades: List[Trade],
        initial_capital: float,
        final_capital: float,
    ) -> Dict[str, float]:
        """Compute overall backtest metrics."""
        if not trades:
            return {
                "total_trades": 0,
                "total_pnl": 0.0,
                "roi": 0.0,
                "win_rate": 0.0,
                "sharpe": 0.0,
                "avg_pnl_per_trade": 0.0,
            }

        total_pnl = sum(t.pnl for t in trades)
        wins = len([t for t in trades if t.pnl > 0])
        win_rate = wins / len(trades)

        # Sharpe ratio
        returns = [t.roi for t in trades]
        sharpe = np.mean(returns) / np.std(returns) * np.sqrt(len(returns)) if np.std(returns) > 0 else 0

        return {
            "total_trades": len(trades),
            "total_pnl": float(total_pnl),
            "roi": float((final_capital - initial_capital) / initial_capital),
            "win_rate": float(win_rate),
            "sharpe": float(sharpe),
            "avg_pnl_per_trade": float(total_pnl / len(trades)),
            "final_capital": float(final_capital),
        }


def save_backtest_result_v3(result: BacktestResult, output_dir: Path) -> None:
    """Save backtest results to JSON files."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save main results
    data = {
        "predictions": [asdict(p) for p in result.predictions],
        "trades": [asdict(t) for t in result.trades],
        "horizon_metrics": {
            str(horizon): asdict(metrics)
            for horizon, metrics in result.horizon_metrics.items()
        },
        "overall_metrics": result.overall_metrics,
        "metadata": result.metadata,
    }

    (output_dir / "backtest_results.json").write_text(json.dumps(data, indent=2, default=str))

    # Save predictions as CSV for easy analysis
    if result.predictions:
        pred_df = pd.DataFrame([asdict(p) for p in result.predictions])
        pred_df.to_csv(output_dir / "predictions.csv", index=False)

    # Save trades as CSV
    if result.trades:
        trade_df = pd.DataFrame([asdict(t) for t in result.trades])
        trade_df.to_csv(output_dir / "trades.csv", index=False)

    # Save horizon metrics as CSV
    if result.horizon_metrics:
        horizon_df = pd.DataFrame([asdict(m) for m in result.horizon_metrics.values()])
        horizon_df.to_csv(output_dir / "horizon_metrics.csv", index=False)
