"""
Backtester for Kalshi earnings mention contracts.

Adapted from fomc_analysis.backtester_v3 for earnings calls.

This backtester:
1. Makes predictions about word mentions in upcoming earnings calls
2. Trades Kalshi YES/NO contracts based on edge over market price
3. Uses actual Kalshi contract outcomes for P&L calculation
4. Provides walk-forward validation with no-lookahead guarantee
5. Optionally applies microstructure calibration and execution simulation
"""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional, Dict, TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from ..microstructure.calibration import KalshiCalibrationCurve
    from ..microstructure.execution import ExecutionSimulator, SpreadFilter


@dataclass
class EarningsPrediction:
    """A prediction for an earnings call mention contract."""
    ticker: str
    call_date: str
    contract: str  # Word being tracked
    predicted_probability: float
    confidence_lower: float
    confidence_upper: float
    actual_outcome: Optional[int] = None  # 1 if mentioned >= threshold, 0 otherwise
    market_price: Optional[float] = None
    edge: Optional[float] = None
    correct: Optional[bool] = None


@dataclass
class Trade:
    """Record of a Kalshi contract trade."""
    ticker: str
    call_date: str
    contract: str
    side: str  # "YES" or "NO"
    position_size: float  # Dollars invested
    entry_price: float  # Price paid (0-1)
    predicted_probability: float
    edge: float
    actual_outcome: int
    pnl: float
    roi: float


@dataclass
class BacktestMetrics:
    """Backtest performance metrics."""
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
    brier_score: float


@dataclass
class BacktestResult:
    """Complete backtest results."""
    predictions: List[EarningsPrediction]
    trades: List[Trade]
    metrics: BacktestMetrics
    metadata: Dict


class EarningsKalshiBacktester:
    """
    Backtest Kalshi earnings mention contract trading.

    Parameters
    ----------
    features : pd.DataFrame
        Features dataframe (index = call_date, columns = feature names)
        Must include word count columns matching contract names
    outcomes : pd.DataFrame
        Kalshi contract outcomes (index = call_date, columns = contract names)
        Values should be 0 or 1 (NO or YES)
    model_class : type
        Model class to use for predictions
    model_params : dict
        Parameters for model initialization
    edge_threshold : float
        Minimum edge required to trade (default: 0.12)
    position_size_pct : float
        Fraction of capital per trade (default: 0.03)
    fee_rate : float
        Kalshi fee rate on profits (default: 0.07)
    transaction_cost_rate : float
        Additional transaction cost (default: 0.01)
    slippage : float
        Price slippage (default: 0.02)
    min_train_window : int
        Minimum historical calls for training (default: 4)
    """

    def __init__(
        self,
        features: pd.DataFrame,
        outcomes: pd.DataFrame,
        model_class: type,
        model_params: dict = None,
        edge_threshold: float = 0.12,
        position_size_pct: float = 0.03,
        fee_rate: float = 0.07,
        transaction_cost_rate: float = 0.01,
        slippage: float = 0.02,
        min_train_window: int = 4,
        yes_edge_threshold: Optional[float] = 0.22,
        no_edge_threshold: Optional[float] = 0.08,
        min_yes_probability: float = 0.65,
        max_no_probability: float = 0.35,
        calibration_curve: Optional["KalshiCalibrationCurve"] = None,
        execution_simulator: Optional["ExecutionSimulator"] = None,
        spread_filter: Optional["SpreadFilter"] = None,
        require_variation: bool = True,
    ):
        self.features = features.sort_index()
        self.outcomes = outcomes.sort_index()
        self.model_class = model_class
        self.model_params = model_params or {}
        self.edge_threshold = edge_threshold
        self.position_size_pct = position_size_pct
        self.fee_rate = fee_rate
        self.transaction_cost_rate = transaction_cost_rate
        self.slippage = slippage
        self.min_train_window = min_train_window
        self.yes_edge_threshold = yes_edge_threshold
        self.no_edge_threshold = no_edge_threshold
        self.min_yes_probability = min_yes_probability
        self.max_no_probability = max_no_probability
        self.calibration_curve = calibration_curve
        self.execution_simulator = execution_simulator
        self.spread_filter = spread_filter
        self.require_variation = require_variation

    def run(
        self,
        ticker: str,
        initial_capital: float = 10000.0,
        market_prices: Optional[pd.DataFrame] = None,
    ) -> BacktestResult:
        """
        Run walk-forward backtest.

        Parameters
        ----------
        ticker : str
            Company stock ticker
        initial_capital : float
            Starting capital
        market_prices : Optional[pd.DataFrame]
            Historical Kalshi market prices (if available)
            Index = call_date, columns = contract names, values = 0-1 prices

        Returns
        -------
        BacktestResult
            Complete backtest results
        """
        capital = initial_capital
        predictions = []
        trades = []

        # Get earnings call dates
        call_dates = sorted(self.features.index)

        # Get contract names from outcomes
        contracts = list(self.outcomes.columns)

        # Walk forward through earnings calls
        for i, current_date in enumerate(call_dates):
            # Skip if not enough historical data
            if i < self.min_train_window:
                continue

            # Train on all previous calls
            train_features = self.features.iloc[:i]
            train_outcomes = self.outcomes.iloc[:i]

            # Fit model for each contract
            for contract in contracts:
                # Get training data for this contract
                y_train = train_outcomes[contract]

                # Skip if no variation in training data (configurable)
                if self.require_variation and y_train.nunique() < 2:
                    continue

                # Fit model
                model = self.model_class(**self.model_params)
                model.fit(train_features, y_train)

                # Make prediction
                current_features = self.features.loc[[current_date]]
                pred = model.predict(current_features)

                predicted_prob = float(pred.iloc[0]["probability"])
                lower_bound = float(pred.iloc[0]["lower_bound"])
                upper_bound = float(pred.iloc[0]["upper_bound"])

                # Get actual outcome
                actual_outcome = int(self.outcomes.loc[current_date, contract])

                # Get market price (if available)
                market_price = None
                if market_prices is not None and contract in market_prices.columns:
                    try:
                        market_price = float(market_prices.loc[current_date, contract])
                    except (KeyError, ValueError):
                        pass

                # If no market price, use simple random baseline (50%)
                if market_price is None:
                    market_price = 0.5

                # Market price in cents (used by calibration and spread filter)
                market_cents = max(1, min(99, int(round(market_price * 100))))

                # Calculate edge — use direction-aware calibrated edge when available
                if self.calibration_curve is not None:
                    # yes_adjusted_edge bakes in the YES overpricing penalty:
                    #   yes_edge = model_prob - (calibrated_prob + yes_penalty)
                    #   no_edge  = (1-model_prob) - (1 - calibrated_prob - yes_penalty)
                    # The penalty makes YES harder to trigger (need stronger signal)
                    # and NO easier to trigger (structural house advantage)
                    yes_edge, _ = self.calibration_curve.yes_adjusted_edge(
                        predicted_prob, market_cents
                    )
                    # yes_edge > 0 → model sees YES value despite penalty
                    # yes_edge < 0 → NO side has edge (penalty works in our favor)
                    edge = yes_edge
                else:
                    edge = predicted_prob - market_price

                # Determine if prediction was correct
                predicted_yes = predicted_prob > 0.5
                correct = (predicted_yes and actual_outcome == 1) or (not predicted_yes and actual_outcome == 0)

                # Create prediction
                prediction = EarningsPrediction(
                    ticker=ticker,
                    call_date=str(current_date),
                    contract=contract,
                    predicted_probability=predicted_prob,
                    confidence_lower=lower_bound,
                    confidence_upper=upper_bound,
                    actual_outcome=actual_outcome,
                    market_price=market_price,
                    edge=edge,
                    correct=correct,
                )
                predictions.append(prediction)

                # Trading logic (same as FOMC backtester_v3)
                if edge > 0:
                    side = "YES"
                    if predicted_prob < self.min_yes_probability:
                        continue
                    required_edge = self.yes_edge_threshold if self.yes_edge_threshold else self.edge_threshold
                    if edge < required_edge:
                        continue
                    raw_entry_price = market_price
                elif edge < 0:
                    side = "NO"
                    if predicted_prob > self.max_no_probability:
                        continue
                    required_edge = self.no_edge_threshold if self.no_edge_threshold else self.edge_threshold
                    if abs(edge) < required_edge:
                        continue
                    raw_entry_price = 1 - market_price
                else:
                    continue

                # Confluence filter: only trade when informational and
                # structural edges agree (like the house, we want
                # every trade to have compounding structural advantage)
                if self.calibration_curve is not None:
                    from ..microstructure.calibration import directional_bias_score
                    bias = directional_bias_score(predicted_prob, market_cents)
                    # YES trade needs positive bias (info + structure agree)
                    # NO trade needs negative bias (info + structure agree)
                    if side == "YES" and bias <= 0:
                        continue
                    if side == "NO" and bias >= 0:
                        continue

                # Spread filter: reject trades where edge doesn't overcome
                # spread cost. Spread varies by price level — tighter at
                # extremes (high-confidence outcomes), wider at mid-range.
                if self.spread_filter is not None:
                    if market_cents <= 15 or market_cents >= 85:
                        estimated_spread = 2  # tight: confident outcomes
                    elif market_cents <= 30 or market_cents >= 70:
                        estimated_spread = 4  # moderate
                    else:
                        estimated_spread = 6  # wide: uncertain outcomes
                    bid_cents = max(1, market_cents - estimated_spread // 2)
                    ask_cents = min(99, market_cents + estimated_spread // 2)
                    if not self.spread_filter.should_trade(edge, bid_cents, ask_cents):
                        continue

                # Adjust for slippage / execution simulation
                if self.execution_simulator is not None:
                    entry_price = self.execution_simulator.adjust_backtest_entry_price(
                        raw_entry_price, side,
                        market_price_cents=int(round(market_price * 100)),
                    )
                else:
                    entry_price = np.clip(raw_entry_price + self.slippage, 0.01, 0.99)

                # Position sizing — when microstructure is active, scale by
                # edge magnitude so larger edges get proportionally larger bets
                # (like a casino betting more on games with higher house edge)
                base_size = capital * self.position_size_pct
                if self.calibration_curve is not None:
                    # Scale position by how much edge exceeds the threshold.
                    # Threshold edge → 1.0x, double threshold → 1.5x, capped at 2x.
                    edge_ratio = abs(edge) / required_edge
                    edge_scaler = min(2.0, 0.5 + 0.5 * edge_ratio)
                    base_size *= edge_scaler
                position_size = base_size
                if position_size <= 0:
                    continue

                # Calculate P&L
                if side == "YES":
                    if actual_outcome == 1:
                        gross_pnl = position_size * (1 - entry_price) / entry_price
                        fees = gross_pnl * self.fee_rate
                        pnl = gross_pnl - fees
                    else:
                        pnl = -position_size
                else:  # NO
                    if actual_outcome == 0:
                        gross_pnl = position_size * entry_price / (1 - entry_price)
                        fees = gross_pnl * self.fee_rate
                        pnl = gross_pnl - fees
                    else:
                        pnl = -position_size

                # Transaction costs
                pnl -= position_size * self.transaction_cost_rate

                # Update capital
                capital += pnl
                roi = pnl / position_size if position_size > 0 else 0

                # Record trade
                trade = Trade(
                    ticker=ticker,
                    call_date=str(current_date),
                    contract=contract,
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

        # Calculate metrics
        metrics = self._calculate_metrics(predictions, trades, initial_capital, capital)

        # Create result
        result = BacktestResult(
            predictions=predictions,
            trades=trades,
            metrics=metrics,
            metadata={
                "ticker": ticker,
                "model_class": self.model_class.__name__,
                "model_params": self.model_params,
                "initial_capital": initial_capital,
                "final_capital": capital,
                "edge_threshold": self.edge_threshold,
                "position_size_pct": self.position_size_pct,
                "fee_rate": self.fee_rate,
                "microstructure": {
                    "calibration_enabled": self.calibration_curve is not None,
                    "calibration_gamma": (
                        self.calibration_curve.gamma
                        if self.calibration_curve is not None else None
                    ),
                    "execution_simulator_enabled": self.execution_simulator is not None,
                    "execution_mode": (
                        self.execution_simulator.mode.value
                        if self.execution_simulator is not None else None
                    ),
                    "spread_filter_enabled": self.spread_filter is not None,
                    "spread_filter_min_net_edge": (
                        self.spread_filter.min_net_edge
                        if self.spread_filter is not None else None
                    ),
                },
            },
        )

        return result

    def _calculate_metrics(
        self,
        predictions: List[EarningsPrediction],
        trades: List[Trade],
        initial_capital: float,
        final_capital: float,
    ) -> BacktestMetrics:
        """Calculate backtest metrics."""
        # Prediction metrics
        total_predictions = len(predictions)
        correct_predictions = sum(1 for p in predictions if p.correct)
        accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0

        # Trading metrics
        total_trades = len(trades)
        winning_trades = sum(1 for t in trades if t.pnl > 0)
        win_rate = winning_trades / total_trades if total_trades > 0 else 0

        total_pnl = sum(t.pnl for t in trades)
        avg_pnl = total_pnl / total_trades if total_trades > 0 else 0

        roi = (final_capital - initial_capital) / initial_capital if initial_capital > 0 else 0

        # Sharpe ratio
        if trades:
            returns = [t.roi for t in trades]
            sharpe = np.mean(returns) / np.std(returns) * np.sqrt(len(returns)) if np.std(returns) > 0 else 0
        else:
            sharpe = 0

        # Brier score (calibration)
        brier_scores = [
            (p.predicted_probability - p.actual_outcome) ** 2
            for p in predictions
            if p.actual_outcome is not None
        ]
        brier_score = np.mean(brier_scores) if brier_scores else 0

        return BacktestMetrics(
            total_predictions=total_predictions,
            correct_predictions=correct_predictions,
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


def save_earnings_backtest_result(result: BacktestResult, output_dir: Path):
    """Save backtest results to files."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save main results
    data = {
        "predictions": [asdict(p) for p in result.predictions],
        "trades": [asdict(t) for t in result.trades],
        "metrics": asdict(result.metrics),
        "metadata": result.metadata,
    }

    (output_dir / "backtest_results.json").write_text(
        json.dumps(data, indent=2, default=str)
    )

    # Save predictions CSV
    if result.predictions:
        pred_df = pd.DataFrame([asdict(p) for p in result.predictions])
        pred_df.to_csv(output_dir / "predictions.csv", index=False)

    # Save trades CSV
    if result.trades:
        trade_df = pd.DataFrame([asdict(t) for t in result.trades])
        trade_df.to_csv(output_dir / "trades.csv", index=False)

    print(f"Results saved to {output_dir}")


def compute_backtest_significance(result: BacktestResult) -> Dict:
    """
    Compute statistical significance of backtest results.

    Uses microstructure statistical tests to determine if the observed
    edge is real or likely due to noise.

    Parameters
    ----------
    result : BacktestResult
        Completed backtest results.

    Returns
    -------
    dict
        Statistical test results including significance, effect size,
        YES/NO asymmetry, and Brier decomposition.
    """
    from ..microstructure.statistical_tests import (
        test_edge_significance,
        test_yes_no_asymmetry,
        test_calibration,
        compute_brier_decomposition,
    )

    stats = {}

    # Overall edge significance
    if result.trades:
        returns = [t.roi for t in result.trades]
        stats["edge_significance"] = test_edge_significance(returns)

        # YES vs NO asymmetry
        yes_returns = [t.roi for t in result.trades if t.side == "YES"]
        no_returns = [t.roi for t in result.trades if t.side == "NO"]
        if yes_returns and no_returns:
            stats["yes_no_asymmetry"] = test_yes_no_asymmetry(yes_returns, no_returns)

    # Calibration
    if result.predictions:
        preds = [p.predicted_probability for p in result.predictions if p.actual_outcome is not None]
        outcomes = [p.actual_outcome for p in result.predictions if p.actual_outcome is not None]
        if preds and outcomes:
            stats["calibration"] = test_calibration(preds, outcomes)
            stats["brier_decomposition"] = compute_brier_decomposition(preds, outcomes)

    return stats


def create_market_prices_from_tracker(
    ticker: str,
    call_dates: List[str],
    price_tracker=None,
    price_data_dir: Optional[Path] = None,
    lookback_days: int = 7,
) -> pd.DataFrame:
    """
    Create market_prices DataFrame from historical price tracker data.

    This function retrieves historical prices for each call date, using the
    price that was available N days before the call (to avoid look-ahead bias).

    Parameters
    ----------
    ticker : str
        Company stock ticker
    call_dates : List[str]
        List of earnings call dates (YYYY-MM-DD format)
    price_tracker : KalshiPriceTracker, optional
        Price tracker instance (creates one if not provided)
    price_data_dir : Path, optional
        Directory with price history data
    lookback_days : int
        Number of days before call date to fetch prices (default: 7)
        This simulates when you would place the trade.

    Returns
    -------
    pd.DataFrame
        Index = call_date, Columns = word names, Values = prices (0-1)
        Compatible with EarningsKalshiBacktester.run(market_prices=...)
    """
    from ..fetchers.kalshi_price_tracker import KalshiPriceTracker

    if price_tracker is None:
        price_tracker = KalshiPriceTracker(data_dir=price_data_dir)

    # Get all historical prices
    history_df = price_tracker.get_price_history(ticker)

    if history_df.empty:
        print(f"No price history found for {ticker}")
        return pd.DataFrame()

    # Build market prices DataFrame
    data = {}

    for call_date in call_dates:
        call_dt = pd.to_datetime(call_date)
        # Get prices N days before the call
        target_date = call_dt - pd.Timedelta(days=lookback_days)

        # Get prices as of target date
        prices = price_tracker.get_historical_prices_for_backtest(
            ticker, target_date.strftime("%Y-%m-%d")
        )

        if prices:
            data[call_date] = prices

    if not data:
        return pd.DataFrame()

    df = pd.DataFrame.from_dict(data, orient="index")
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()

    return df


def run_backtest_with_historical_prices(
    ticker: str,
    features: pd.DataFrame,
    outcomes: pd.DataFrame,
    model_class: type,
    model_params: dict = None,
    price_data_dir: Optional[Path] = None,
    lookback_days: int = 7,
    **backtester_kwargs,
) -> BacktestResult:
    """
    Run backtest using historical price tracker data.

    This is a convenience function that:
    1. Creates market_prices DataFrame from price tracker
    2. Runs the backtest with those prices

    Parameters
    ----------
    ticker : str
        Company stock ticker
    features : pd.DataFrame
        Features dataframe (index = call_date, columns = feature names)
    outcomes : pd.DataFrame
        Contract outcomes (index = call_date, columns = contract names)
    model_class : type
        Model class for predictions
    model_params : dict, optional
        Parameters for model initialization
    price_data_dir : Path, optional
        Directory with price history data
    lookback_days : int
        Days before call to fetch prices (default: 7)
    **backtester_kwargs
        Additional arguments for EarningsKalshiBacktester

    Returns
    -------
    BacktestResult
        Complete backtest results with historical prices
    """
    # Get call dates from features
    call_dates = [str(d.date()) if hasattr(d, 'date') else str(d)
                  for d in features.index]

    # Create market prices from tracker
    market_prices = create_market_prices_from_tracker(
        ticker=ticker,
        call_dates=call_dates,
        price_data_dir=price_data_dir,
        lookback_days=lookback_days,
    )

    if market_prices.empty:
        print(f"Warning: No historical prices found for {ticker}. "
              "Using default 0.5 prices.")

    # Create and run backtester
    backtester = EarningsKalshiBacktester(
        features=features,
        outcomes=outcomes,
        model_class=model_class,
        model_params=model_params,
        **backtester_kwargs,
    )

    result = backtester.run(
        ticker=ticker,
        market_prices=market_prices if not market_prices.empty else None,
    )

    # Add price metadata
    result.metadata["price_lookback_days"] = lookback_days
    result.metadata["historical_prices_used"] = not market_prices.empty

    return result


def create_microstructure_backtester(
    features: pd.DataFrame,
    outcomes: pd.DataFrame,
    model_class: type,
    model_params: dict = None,
    execution_mode: str = "hybrid",
    spread_filter_min_net_edge: float = 0.03,
    spread_filter_max_spread: int = 15,
    calibration_gamma: float = 1.064,
    calibration_yes_penalty: float = 0.0048,
    **backtester_kwargs,
) -> EarningsKalshiBacktester:
    """
    Create a backtester with all microstructure components pre-wired.

    This is the recommended way to create a backtester that uses calibration,
    execution simulation, and spread filtering from the microstructure research.

    Parameters
    ----------
    features : pd.DataFrame
        Features dataframe (index = call_date, columns = feature names).
    outcomes : pd.DataFrame
        Contract outcomes (index = call_date, columns = contract names).
    model_class : type
        Model class for predictions.
    model_params : dict, optional
        Model initialization parameters.
    execution_mode : str
        Execution mode: "taker", "maker", or "hybrid" (default: "hybrid").
    spread_filter_min_net_edge : float
        Minimum edge after spread cost to trade (default: 0.03).
    spread_filter_max_spread : int
        Maximum spread in cents to trade (default: 15).
    calibration_gamma : float
        Logit-scaling parameter for calibration (default: 1.064).
    calibration_yes_penalty : float
        YES-side overpricing penalty (default: 0.0048).
    **backtester_kwargs
        Additional arguments for EarningsKalshiBacktester (e.g.,
        edge_threshold, position_size_pct, fee_rate).

    Returns
    -------
    EarningsKalshiBacktester
        Backtester with microstructure components configured.
    """
    from ..microstructure.calibration import KalshiCalibrationCurve
    from ..microstructure.execution import ExecutionSimulator, SpreadFilter, ExecutionMode

    mode_map = {
        "taker": ExecutionMode.TAKER,
        "maker": ExecutionMode.MAKER,
        "hybrid": ExecutionMode.HYBRID,
    }

    calibration_curve = KalshiCalibrationCurve(
        gamma=calibration_gamma,
        yes_penalty=calibration_yes_penalty,
    )

    execution_simulator = ExecutionSimulator(
        mode=mode_map.get(execution_mode, ExecutionMode.HYBRID),
    )

    spread_filter = SpreadFilter(
        min_net_edge=spread_filter_min_net_edge,
        max_spread_cents=spread_filter_max_spread,
    )

    return EarningsKalshiBacktester(
        features=features,
        outcomes=outcomes,
        model_class=model_class,
        model_params=model_params,
        calibration_curve=calibration_curve,
        execution_simulator=execution_simulator,
        spread_filter=spread_filter,
        **backtester_kwargs,
    )
