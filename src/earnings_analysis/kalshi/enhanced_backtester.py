"""
Enhanced backtester with advanced trading optimizations.

Key improvements over base backtester:
1. Kelly Criterion position sizing (optimal bet sizing)
2. Confidence-adjusted positions (narrower CI = larger position)
3. Correlation-aware exposure limits
4. Calibration tracking and adjustment
5. Maximum drawdown protection
6. Time-based features (earnings timing patterns)
7. Microstructure calibration and execution simulation
"""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict, field
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Dict, Set, TYPE_CHECKING
import warnings

import numpy as np
import pandas as pd

from .backtester import (
    EarningsPrediction,
    Trade,
    BacktestMetrics,
    BacktestResult,
)

if TYPE_CHECKING:
    from ..microstructure.calibration import KalshiCalibrationCurve
    from ..microstructure.execution import ExecutionSimulator


@dataclass
class EnhancedTrade(Trade):
    """Trade with additional optimization metadata."""
    kelly_fraction: float = 0.0
    confidence_multiplier: float = 1.0
    correlation_penalty: float = 1.0
    calibration_adjustment: float = 0.0


@dataclass
class EnhancedMetrics(BacktestMetrics):
    """Extended metrics for enhanced backtester."""
    max_drawdown: float = 0.0
    calmar_ratio: float = 0.0  # Return / Max Drawdown
    avg_kelly_fraction: float = 0.0
    calibration_error: float = 0.0
    correlation_exposure_avg: float = 0.0


@dataclass
class CalibrationBin:
    """Track calibration within a probability bin."""
    bin_start: float
    bin_end: float
    predictions: int = 0
    actual_positives: int = 0

    @property
    def predicted_rate(self) -> float:
        return (self.bin_start + self.bin_end) / 2

    @property
    def actual_rate(self) -> float:
        return self.actual_positives / self.predictions if self.predictions > 0 else 0

    @property
    def calibration_error(self) -> float:
        return self.actual_rate - self.predicted_rate


class EnhancedEarningsBacktester:
    """
    Enhanced backtester with advanced trading optimizations.

    Key Features:
    - Kelly criterion position sizing for optimal growth
    - Confidence-adjusted sizing based on credible interval width
    - Correlation-aware exposure management
    - Dynamic calibration adjustment
    - Maximum drawdown protection

    Parameters
    ----------
    features : pd.DataFrame
        Features dataframe (index = call_date, columns = feature names)
    outcomes : pd.DataFrame
        Contract outcomes (index = call_date, columns = contract names)
    model_class : type
        Model class for predictions
    model_params : dict
        Model initialization parameters
    kelly_fraction : float
        Fraction of Kelly to bet (default: 0.25 = quarter Kelly for safety)
    max_position_pct : float
        Maximum position size as fraction of capital (default: 0.10)
    min_position_pct : float
        Minimum position size to execute (default: 0.01)
    confidence_scaling : bool
        Scale position by model confidence (default: True)
    correlation_limit : float
        Maximum total exposure to correlated contracts (default: 0.25)
    max_drawdown_limit : float
        Stop trading if drawdown exceeds this (default: 0.20)
    calibration_window : int
        Number of predictions for rolling calibration (default: 50)
    edge_threshold : float
        Minimum edge to trade (default: 0.10)
    fee_rate : float
        Kalshi fee rate (default: 0.07)
    slippage : float
        Price slippage (default: 0.02)
    min_train_window : int
        Minimum training calls (default: 4)
    """

    # Word categories for correlation grouping
    WORD_CATEGORIES = {
        "ai_tech": {"ai", "artificial intelligence", "machine learning", "neural", "llama",
                   "grok", "meta ai", "generative ai", "gen ai", "ai recommendation"},
        "vr_ar": {"vr", "virtual reality", "ar", "augmented reality", "metaverse",
                 "quest", "horizon worlds", "orion", "ray-ban"},
        "social": {"tiktok", "instagram", "threads", "whatsapp", "reels", "activitypub"},
        "autonomous": {"robotaxi", "fsd", "full self driving", "autonomous", "waymo", "dojo"},
        "energy": {"battery", "energy", "solar", "powerwall", "megapack", "supercharger"},
        "regulatory": {"regulator", "regulatory", "regulation", "antitrust", "china",
                      "tariff", "election", "europe", "european"},
        "financial": {"revenue", "margin", "dividend", "efficiency", "growth", "demand"},
    }

    def __init__(
        self,
        features: pd.DataFrame,
        outcomes: pd.DataFrame,
        model_class: type,
        model_params: dict = None,
        kelly_fraction: float = 0.25,
        max_position_pct: float = 0.10,
        min_position_pct: float = 0.01,
        confidence_scaling: bool = True,
        correlation_limit: float = 0.25,
        max_drawdown_limit: float = 0.20,
        calibration_window: int = 50,
        edge_threshold: float = 0.10,
        yes_edge_threshold: float = 0.15,
        no_edge_threshold: float = 0.08,
        fee_rate: float = 0.07,
        transaction_cost_rate: float = 0.01,
        slippage: float = 0.02,
        min_train_window: int = 4,
        calibration_curve: Optional["KalshiCalibrationCurve"] = None,
        execution_simulator: Optional["ExecutionSimulator"] = None,
    ):
        self.features = features.sort_index()
        self.outcomes = outcomes.sort_index()
        self.model_class = model_class
        self.model_params = model_params or {}

        # Kelly and position sizing
        self.kelly_fraction = kelly_fraction
        self.max_position_pct = max_position_pct
        self.min_position_pct = min_position_pct
        self.confidence_scaling = confidence_scaling

        # Risk management
        self.correlation_limit = correlation_limit
        self.max_drawdown_limit = max_drawdown_limit

        # Calibration
        self.calibration_window = calibration_window
        self.calibration_bins = self._init_calibration_bins()

        # Trading parameters
        self.edge_threshold = edge_threshold
        self.yes_edge_threshold = yes_edge_threshold
        self.no_edge_threshold = no_edge_threshold
        self.fee_rate = fee_rate
        self.transaction_cost_rate = transaction_cost_rate
        self.slippage = slippage
        self.min_train_window = min_train_window

        # Microstructure
        self.calibration_curve = calibration_curve
        self.execution_simulator = execution_simulator

        # State tracking
        self.recent_predictions: List[EarningsPrediction] = []
        self.current_exposure: Dict[str, float] = {}  # category -> exposure

    def _init_calibration_bins(self) -> List[CalibrationBin]:
        """Initialize calibration bins for tracking."""
        bins = []
        for i in range(10):
            bins.append(CalibrationBin(
                bin_start=i * 0.1,
                bin_end=(i + 1) * 0.1,
            ))
        return bins

    def _get_word_category(self, word: str) -> Optional[str]:
        """Get the category for a word (for correlation tracking)."""
        word_lower = word.lower()
        for category, words in self.WORD_CATEGORIES.items():
            for w in words:
                if w in word_lower or word_lower in w:
                    return category
        return None

    def _calculate_kelly_fraction(
        self,
        predicted_prob: float,
        market_price: float,
        side: str,
    ) -> float:
        """
        Calculate Kelly criterion bet fraction.

        Kelly formula: f* = (bp - q) / b
        where:
            b = odds received on bet (net payout per dollar bet)
            p = probability of winning
            q = probability of losing = 1 - p

        For binary contracts:
        - YES bet at price P: win (1-P)/P, probability p
        - NO bet at price P: win P/(1-P), probability (1-p)
        """
        if side == "YES":
            # Bet at market_price, win (1-market_price)/market_price if YES
            b = (1 - market_price) / market_price if market_price > 0 else 0
            p = predicted_prob
        else:
            # Bet at (1-market_price), win market_price/(1-market_price) if NO
            b = market_price / (1 - market_price) if market_price < 1 else 0
            p = 1 - predicted_prob

        q = 1 - p

        if b <= 0:
            return 0

        kelly = (b * p - q) / b

        # Clamp to valid range
        return max(0, min(1, kelly))

    def _calculate_confidence_multiplier(
        self,
        lower_bound: float,
        upper_bound: float,
    ) -> float:
        """
        Calculate confidence multiplier based on credible interval width.

        Narrower CI = more confident = larger multiplier (up to 1.5x)
        Wider CI = less confident = smaller multiplier (down to 0.5x)
        """
        ci_width = upper_bound - lower_bound

        # Typical CI width is around 0.3-0.5 for moderate certainty
        # Very confident: < 0.2
        # Very uncertain: > 0.6

        if ci_width < 0.15:
            return 1.5  # Very confident
        elif ci_width < 0.25:
            return 1.25
        elif ci_width < 0.35:
            return 1.0  # Normal
        elif ci_width < 0.50:
            return 0.75
        else:
            return 0.5  # Very uncertain

    def _get_correlation_penalty(
        self,
        word: str,
        current_date: str,
    ) -> float:
        """
        Calculate penalty for correlated exposure.

        If we already have exposure to similar words, reduce position size.
        """
        category = self._get_word_category(word)

        if category is None:
            return 1.0  # No correlation tracking for uncategorized words

        current_exposure = self.current_exposure.get(category, 0)

        if current_exposure >= self.correlation_limit:
            return 0.0  # Don't add more to this category

        # Linear penalty as we approach limit
        remaining_capacity = self.correlation_limit - current_exposure
        return min(1.0, remaining_capacity / self.correlation_limit)

    def _update_calibration(self, prediction: EarningsPrediction):
        """Update calibration tracking with new prediction."""
        if prediction.actual_outcome is None:
            return

        # Find appropriate bin
        prob = prediction.predicted_probability
        bin_idx = min(9, int(prob * 10))

        self.calibration_bins[bin_idx].predictions += 1
        if prediction.actual_outcome == 1:
            self.calibration_bins[bin_idx].actual_positives += 1

    def _get_calibration_adjustment(self, predicted_prob: float) -> float:
        """
        Get calibration adjustment for a prediction.

        Returns adjustment to add to predicted probability based on
        historical calibration error in that bin.
        """
        bin_idx = min(9, int(predicted_prob * 10))
        bin_data = self.calibration_bins[bin_idx]

        # Need sufficient data for reliable adjustment
        if bin_data.predictions < 10:
            return 0.0

        # Return the calibration error (positive = we underpredict)
        return bin_data.calibration_error

    def _calculate_max_drawdown(self, equity_curve: List[float]) -> float:
        """Calculate maximum drawdown from equity curve."""
        if len(equity_curve) < 2:
            return 0.0

        peak = equity_curve[0]
        max_dd = 0.0

        for value in equity_curve:
            if value > peak:
                peak = value
            dd = (peak - value) / peak if peak > 0 else 0
            max_dd = max(max_dd, dd)

        return max_dd

    def run(
        self,
        ticker: str,
        initial_capital: float = 10000.0,
        market_prices: Optional[pd.DataFrame] = None,
    ) -> BacktestResult:
        """
        Run enhanced walk-forward backtest.

        Parameters
        ----------
        ticker : str
            Company ticker symbol
        initial_capital : float
            Starting capital
        market_prices : Optional[pd.DataFrame]
            Historical market prices if available

        Returns
        -------
        BacktestResult
            Complete backtest results with enhanced metrics
        """
        capital = initial_capital
        peak_capital = initial_capital
        predictions = []
        trades = []
        equity_curve = [initial_capital]

        # Reset state
        self.recent_predictions = []
        self.current_exposure = {}
        self.calibration_bins = self._init_calibration_bins()

        call_dates = sorted(self.features.index)
        contracts = list(self.outcomes.columns)

        trading_halted = False

        for i, current_date in enumerate(call_dates):
            if i < self.min_train_window:
                continue

            # Check drawdown limit
            current_dd = (peak_capital - capital) / peak_capital if peak_capital > 0 else 0
            if current_dd > self.max_drawdown_limit:
                if not trading_halted:
                    warnings.warn(f"Trading halted at {current_date}: drawdown {current_dd:.1%} > limit {self.max_drawdown_limit:.1%}")
                    trading_halted = True

            # Reset per-call exposure tracking
            call_exposure: Dict[str, float] = {}

            train_features = self.features.iloc[:i]
            train_outcomes = self.outcomes.iloc[:i]

            for contract in contracts:
                y_train = train_outcomes[contract]

                if y_train.nunique() < 2:
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

                # Get calibration adjustment
                cal_adj = self._get_calibration_adjustment(predicted_prob)
                adjusted_prob = np.clip(predicted_prob + cal_adj, 0.01, 0.99)

                actual_outcome = int(self.outcomes.loc[current_date, contract])

                # Get market price
                market_price = 0.5
                if market_prices is not None and contract in market_prices.columns:
                    try:
                        market_price = float(market_prices.loc[current_date, contract])
                    except (KeyError, ValueError):
                        pass

                # Calculate edge — use microstructure calibration when available
                if self.calibration_curve is not None:
                    market_cents = int(round(market_price * 100))
                    market_cents = max(1, min(99, market_cents))
                    edge = self.calibration_curve.calibrated_edge(
                        adjusted_prob, market_cents
                    )
                else:
                    edge = adjusted_prob - market_price

                predicted_yes = adjusted_prob > 0.5
                correct = (predicted_yes and actual_outcome == 1) or (not predicted_yes and actual_outcome == 0)

                prediction = EarningsPrediction(
                    ticker=ticker,
                    call_date=str(current_date),
                    contract=contract,
                    predicted_probability=adjusted_prob,
                    confidence_lower=lower_bound,
                    confidence_upper=upper_bound,
                    actual_outcome=actual_outcome,
                    market_price=market_price,
                    edge=edge,
                    correct=correct,
                )
                predictions.append(prediction)

                # Update calibration tracking
                self._update_calibration(prediction)

                # Skip trading if halted
                if trading_halted:
                    continue

                # Determine trade direction
                if edge > 0:
                    side = "YES"
                    required_edge = self.yes_edge_threshold
                    if edge < required_edge:
                        continue
                    raw_entry_price = market_price
                elif edge < 0:
                    side = "NO"
                    required_edge = self.no_edge_threshold
                    if abs(edge) < required_edge:
                        continue
                    raw_entry_price = 1 - market_price
                else:
                    continue

                # Calculate optimal position size
                kelly = self._calculate_kelly_fraction(adjusted_prob, market_price, side)

                # Apply Kelly fraction (e.g., quarter Kelly)
                kelly_position = kelly * self.kelly_fraction

                # Apply confidence scaling
                conf_mult = 1.0
                if self.confidence_scaling:
                    conf_mult = self._calculate_confidence_multiplier(lower_bound, upper_bound)

                # Apply correlation penalty
                corr_penalty = self._get_correlation_penalty(contract, str(current_date))

                # Final position size
                position_pct = kelly_position * conf_mult * corr_penalty
                position_pct = np.clip(position_pct, 0, self.max_position_pct)

                if position_pct < self.min_position_pct:
                    continue

                position_size = capital * position_pct

                # Update exposure tracking
                category = self._get_word_category(contract)
                if category:
                    call_exposure[category] = call_exposure.get(category, 0) + position_pct

                # Adjust for slippage / execution simulation
                if self.execution_simulator is not None:
                    entry_price = self.execution_simulator.adjust_backtest_entry_price(
                        raw_entry_price, side,
                        market_price_cents=int(round(market_price * 100)),
                    )
                else:
                    entry_price = np.clip(raw_entry_price + self.slippage, 0.01, 0.99)

                # Directional sizing: NO side gets structural edge bonus
                if self.calibration_curve is not None:
                    if side == "NO":
                        position_pct *= 1.15
                    else:
                        position_pct *= 0.90
                    position_pct = np.clip(position_pct, 0, self.max_position_pct)

                # Calculate P&L
                if side == "YES":
                    if actual_outcome == 1:
                        gross_pnl = position_size * (1 - entry_price) / entry_price
                        fees = gross_pnl * self.fee_rate
                        pnl = gross_pnl - fees
                    else:
                        pnl = -position_size
                else:
                    if actual_outcome == 0:
                        gross_pnl = position_size * entry_price / (1 - entry_price)
                        fees = gross_pnl * self.fee_rate
                        pnl = gross_pnl - fees
                    else:
                        pnl = -position_size

                pnl -= position_size * self.transaction_cost_rate

                capital += pnl
                peak_capital = max(peak_capital, capital)
                roi = pnl / position_size if position_size > 0 else 0

                trade = EnhancedTrade(
                    ticker=ticker,
                    call_date=str(current_date),
                    contract=contract,
                    side=side,
                    position_size=position_size,
                    entry_price=entry_price,
                    predicted_probability=adjusted_prob,
                    edge=edge,
                    actual_outcome=actual_outcome,
                    pnl=pnl,
                    roi=roi,
                    kelly_fraction=kelly,
                    confidence_multiplier=conf_mult,
                    correlation_penalty=corr_penalty,
                    calibration_adjustment=cal_adj,
                )
                trades.append(trade)

            # Update global exposure after processing all contracts for this call
            self.current_exposure = call_exposure
            equity_curve.append(capital)

        # Calculate enhanced metrics
        metrics = self._calculate_enhanced_metrics(
            predictions, trades, initial_capital, capital, equity_curve
        )

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
                "kelly_fraction": self.kelly_fraction,
                "max_position_pct": self.max_position_pct,
                "confidence_scaling": self.confidence_scaling,
                "correlation_limit": self.correlation_limit,
                "max_drawdown_limit": self.max_drawdown_limit,
                "trading_halted": trading_halted,
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
                },
                "calibration_bins": [
                    {"bin": f"{b.bin_start:.1f}-{b.bin_end:.1f}",
                     "predictions": b.predictions,
                     "actual_rate": b.actual_rate,
                     "error": b.calibration_error}
                    for b in self.calibration_bins if b.predictions > 0
                ],
            },
        )

        return result

    def _calculate_enhanced_metrics(
        self,
        predictions: List[EarningsPrediction],
        trades: List[EnhancedTrade],
        initial_capital: float,
        final_capital: float,
        equity_curve: List[float],
    ) -> EnhancedMetrics:
        """Calculate enhanced backtest metrics."""
        # Base metrics
        total_predictions = len(predictions)
        correct_predictions = sum(1 for p in predictions if p.correct)
        accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0

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

        # Brier score
        brier_scores = [
            (p.predicted_probability - p.actual_outcome) ** 2
            for p in predictions if p.actual_outcome is not None
        ]
        brier_score = np.mean(brier_scores) if brier_scores else 0

        # Enhanced metrics
        max_dd = self._calculate_max_drawdown(equity_curve)
        calmar = roi / max_dd if max_dd > 0 else 0

        avg_kelly = np.mean([t.kelly_fraction for t in trades]) if trades else 0

        # Calibration error (mean absolute error across bins)
        cal_errors = [abs(b.calibration_error) for b in self.calibration_bins if b.predictions >= 5]
        cal_error = np.mean(cal_errors) if cal_errors else 0

        # Correlation exposure
        corr_exposures = [t.correlation_penalty for t in trades]
        avg_corr_exposure = 1 - np.mean(corr_exposures) if corr_exposures else 0

        return EnhancedMetrics(
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
            max_drawdown=max_dd,
            calmar_ratio=calmar,
            avg_kelly_fraction=avg_kelly,
            calibration_error=cal_error,
            correlation_exposure_avg=avg_corr_exposure,
        )


def save_enhanced_backtest_result(result: BacktestResult, output_dir: Path):
    """Save enhanced backtest results."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    data = {
        "predictions": [asdict(p) for p in result.predictions],
        "trades": [asdict(t) for t in result.trades],
        "metrics": asdict(result.metrics),
        "metadata": result.metadata,
    }

    (output_dir / "enhanced_backtest_results.json").write_text(
        json.dumps(data, indent=2, default=str)
    )

    if result.predictions:
        pred_df = pd.DataFrame([asdict(p) for p in result.predictions])
        pred_df.to_csv(output_dir / "predictions.csv", index=False)

    if result.trades:
        trade_df = pd.DataFrame([asdict(t) for t in result.trades])
        trade_df.to_csv(output_dir / "trades.csv", index=False)

    print(f"Enhanced results saved to {output_dir}")
