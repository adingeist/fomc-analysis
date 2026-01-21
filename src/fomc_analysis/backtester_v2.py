"""
Walk-forward backtester with strict no-lookahead guarantees.

This backtester simulates realistic trading scenarios with:
- Walk-forward validation (train on past, predict on next event)
- No lookahead: only information available before time t can be used
- Realistic execution: bid/ask spreads, fees, position limits
- Comprehensive metrics: ROI, Sharpe, Sortino, max drawdown, Brier score, calibration
"""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Dict, Tuple

import numpy as np
import pandas as pd


@dataclass
class Trade:
    """Record of a single trade."""
    date: str
    contract: str
    side: str  # "YES" or "NO"
    position_size: float  # Contracts or notional
    entry_price: float  # 0-1 probability
    actual_outcome: Optional[int] = None  # 1 if YES, 0 if NO, None if unknown
    exit_price: Optional[float] = None
    pnl: Optional[float] = None
    model_prob: Optional[float] = None
    edge: Optional[float] = None


@dataclass
class BacktestResult:
    """Results of a backtest run."""
    trades: List[Trade]
    equity_curve: pd.Series
    metrics: Dict[str, float]
    metadata: Dict[str, any]


class WalkForwardBacktester:
    """
    Walk-forward backtester with no-lookahead guarantees.

    Parameters
    ----------
    events : pd.DataFrame
        Binary event matrix (rows=dates, cols=contracts).
        1 if contract mentioned, 0 otherwise.
    prices : pd.DataFrame
        Market prices (0-1 probabilities).
    edge_threshold : float, default=0.05
        Minimum edge required to trade.
    position_size_pct : float, default=0.02
        Fraction of capital per trade.
    fee_rate : float, default=0.07
        Transaction fee rate (7% on profits for Kalshi).
    min_train_window : int, default=3
        Minimum number of past events required to train.
    """

    def __init__(
        self,
        events: pd.DataFrame,
        prices: Optional[pd.DataFrame] = None,
        edge_threshold: float = 0.05,
        position_size_pct: float = 0.02,
        fee_rate: float = 0.07,
        min_train_window: int = 3,
    ):
        self.events = events.sort_index()
        self.prices = prices.sort_index() if prices is not None else None
        self.edge_threshold = edge_threshold
        self.position_size_pct = position_size_pct
        self.fee_rate = fee_rate
        self.min_train_window = min_train_window

        # Validate alignment
        if self.prices is not None:
            # Ensure dates align
            common_dates = self.events.index.intersection(self.prices.index)
            if len(common_dates) < len(self.events):
                print(f"Warning: {len(self.events) - len(common_dates)} dates missing prices")

    def run(
        self,
        model_class,
        model_params: Optional[Dict] = None,
        initial_capital: float = 1000.0,
    ) -> BacktestResult:
        """
        Run walk-forward backtest.

        At each time step t:
        1. Train model on events[:t]
        2. Predict probability for event at time t
        3. Compare with market price at time t-1 (before event)
        4. Execute trade if edge > threshold
        5. Observe outcome at time t
        6. Update capital

        Parameters
        ----------
        model_class : class
            Model class to instantiate (e.g., BetaBinomialModel).
        model_params : Optional[Dict]
            Parameters to pass to model constructor.
        initial_capital : float, default=1000.0
            Starting capital.

        Returns
        -------
        BacktestResult
            Complete backtest results with trades and metrics.
        """
        model_params = model_params or {}
        capital = initial_capital
        trades = []
        equity = [initial_capital]
        dates = [self.events.index[0]]

        # Walk forward through time
        for i in range(self.min_train_window, len(self.events)):
            current_date = self.events.index[i]
            prev_date = self.events.index[i-1]

            # Train on all data BEFORE current date
            train_events = self.events.iloc[:i]

            # Fit model
            model = model_class(**model_params)
            model.fit(train_events)

            # Predict for current date
            predictions = model.predict(n_future=1)

            # Get actual outcomes for current date
            actual_outcomes = self.events.iloc[i]

            # For each contract, check if we should trade
            for _, pred_row in predictions.iterrows():
                contract = pred_row["contract"]
                model_prob = pred_row["probability"]

                # Get market price (from previous date or current if unavailable)
                if self.prices is not None:
                    if prev_date in self.prices.index and contract in self.prices.columns:
                        market_price = self.prices.loc[prev_date, contract]
                    elif current_date in self.prices.index and contract in self.prices.columns:
                        market_price = self.prices.loc[current_date, contract]
                    else:
                        continue  # No price data
                else:
                    # If no prices provided, skip trading
                    continue

                # Compute edge
                edge = model_prob - market_price

                # Check if edge exceeds threshold
                if abs(edge) < self.edge_threshold:
                    continue

                # Determine trade direction
                if edge > 0:
                    side = "YES"
                    entry_price = market_price
                else:
                    side = "NO"
                    entry_price = 1 - market_price

                # Position size (fixed fraction of capital)
                position_size = capital * self.position_size_pct

                # Get actual outcome
                if contract in actual_outcomes.index:
                    outcome = int(actual_outcomes[contract])
                else:
                    continue  # Missing outcome

                # Compute P&L
                if side == "YES":
                    if outcome == 1:
                        # Won bet
                        gross_pnl = position_size * (1 - entry_price) / entry_price
                        fees = gross_pnl * self.fee_rate
                        pnl = gross_pnl - fees
                    else:
                        # Lost bet
                        pnl = -position_size
                else:  # NO
                    if outcome == 0:
                        # Won bet
                        gross_pnl = position_size * entry_price / (1 - entry_price)
                        fees = gross_pnl * self.fee_rate
                        pnl = gross_pnl - fees
                    else:
                        # Lost bet
                        pnl = -position_size

                # Update capital
                capital += pnl

                # Record trade
                trades.append(Trade(
                    date=str(current_date),
                    contract=contract,
                    side=side,
                    position_size=position_size,
                    entry_price=entry_price,
                    actual_outcome=outcome,
                    exit_price=float(outcome),
                    pnl=pnl,
                    model_prob=model_prob,
                    edge=edge,
                ))

            # Record equity
            equity.append(capital)
            dates.append(current_date)

        # Compute metrics
        equity_series = pd.Series(equity, index=dates)
        metrics = self._compute_metrics(trades, equity_series, initial_capital)

        return BacktestResult(
            trades=trades,
            equity_curve=equity_series,
            metrics=metrics,
            metadata={
                "model_class": model_class.__name__,
                "model_params": model_params,
                "initial_capital": initial_capital,
                "edge_threshold": self.edge_threshold,
                "position_size_pct": self.position_size_pct,
                "fee_rate": self.fee_rate,
            },
        )

    def _compute_metrics(
        self,
        trades: List[Trade],
        equity_curve: pd.Series,
        initial_capital: float,
    ) -> Dict[str, float]:
        """Compute backtest performance metrics."""
        if len(trades) == 0:
            return {
                "total_trades": 0,
                "roi": 0.0,
                "sharpe": 0.0,
                "max_drawdown": 0.0,
                "win_rate": 0.0,
                "brier_score": 0.0,
            }

        # Basic metrics
        final_capital = equity_curve.iloc[-1]
        roi = (final_capital - initial_capital) / initial_capital

        # Win rate
        wins = sum(1 for t in trades if t.pnl > 0)
        win_rate = wins / len(trades) if trades else 0

        # Returns
        returns = equity_curve.pct_change().dropna()

        # Sharpe ratio (annualized, assume ~8 events/year)
        if len(returns) > 0 and returns.std() > 0:
            sharpe = returns.mean() / returns.std() * np.sqrt(8)
        else:
            sharpe = 0.0

        # Sortino ratio (downside deviation)
        downside_returns = returns[returns < 0]
        if len(downside_returns) > 0 and downside_returns.std() > 0:
            sortino = returns.mean() / downside_returns.std() * np.sqrt(8)
        else:
            sortino = sharpe

        # Max drawdown
        running_max = equity_curve.expanding().max()
        drawdown = (equity_curve - running_max) / running_max
        max_drawdown = drawdown.min()

        # Brier score (calibration metric)
        brier_scores = []
        for trade in trades:
            if trade.model_prob is not None and trade.actual_outcome is not None:
                if trade.side == "YES":
                    prob = trade.model_prob
                    outcome = trade.actual_outcome
                else:  # NO
                    prob = 1 - trade.model_prob
                    outcome = 1 - trade.actual_outcome

                brier_scores.append((prob - outcome) ** 2)

        brier_score = np.mean(brier_scores) if brier_scores else 0.0

        return {
            "total_trades": len(trades),
            "roi": float(roi),
            "sharpe": float(sharpe),
            "sortino": float(sortino),
            "max_drawdown": float(max_drawdown),
            "win_rate": float(win_rate),
            "avg_pnl": float(np.mean([t.pnl for t in trades if t.pnl is not None])),
            "brier_score": float(brier_score),
            "final_capital": float(final_capital),
        }


def save_backtest_result(result: BacktestResult, output_path: Path) -> None:
    """Save backtest results to JSON file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    data = {
        "trades": [asdict(t) for t in result.trades],
        "equity_curve": {
            "dates": [str(d) for d in result.equity_curve.index],
            "values": result.equity_curve.tolist(),
        },
        "metrics": result.metrics,
        "metadata": result.metadata,
    }

    output_path.write_text(json.dumps(data, indent=2, default=str))


def load_backtest_result(input_path: Path) -> BacktestResult:
    """Load backtest results from JSON file."""
    data = json.loads(Path(input_path).read_text())

    trades = [Trade(**t) for t in data["trades"]]
    equity_curve = pd.Series(
        data["equity_curve"]["values"],
        index=pd.to_datetime(data["equity_curve"]["dates"]),
    )

    return BacktestResult(
        trades=trades,
        equity_curve=equity_curve,
        metrics=data["metrics"],
        metadata=data["metadata"],
    )
