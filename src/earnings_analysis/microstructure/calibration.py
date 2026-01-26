"""
Price-to-win-rate calibration for Kalshi prediction markets.

Empirical findings from Becker (2025) show systematic miscalibration in
Kalshi markets:
- Longshot YES contracts (low prices) are overpriced: actual win rate < implied
- Favorite YES contracts (high prices) are slightly underpriced
- YES contracts are systematically overpriced relative to NO at all levels

This module provides calibration correction to improve edge calculations.

Calibration model:
    logit(actual_prob) = gamma * logit(market_price)

where gamma > 1 compresses longshots and expands favorites. Estimated from:
- Price 5c: actual win rate 4.18% (implied 5.0%)
- Price 95c: actual win rate 95.83% (implied 95.0%)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


# Default logit-scaling parameter estimated from Becker (2025) data.
# logit(0.0418) / logit(0.05) ~ 1.064
_DEFAULT_GAMMA = 1.064

# YES-side overpricing penalty (cents). Becker found makers buying NO earn
# +1.25% excess vs +0.77% buying YES, a ~0.48pp directional gap.
_DEFAULT_YES_PENALTY = 0.0048


def _logit(p: float) -> float:
    """Logit transform, clamped to avoid infinities."""
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return float(np.log(p / (1 - p)))


def _inv_logit(x: float) -> float:
    """Inverse logit (sigmoid)."""
    return float(1.0 / (1.0 + np.exp(-x)))


@dataclass
class KalshiCalibrationCurve:
    """
    Calibration curve mapping market prices to empirical win rates.

    Two modes:
    1. Parametric (default): Uses logit-scaling model from aggregate data
    2. Empirical: Uses a lookup table built from actual trade-level data

    Parameters
    ----------
    gamma : float
        Logit-scaling parameter. >1 means longshots are overpriced.
        Default estimated from Becker (2025).
    yes_penalty : float
        Additional YES-side overpricing (probability points).
        Reflects systematic taker preference for YES contracts.
    empirical_table : dict
        Optional price_cents -> actual_win_rate lookup table.
        When provided, overrides parametric model at those price levels.
    """

    gamma: float = _DEFAULT_GAMMA
    yes_penalty: float = _DEFAULT_YES_PENALTY
    empirical_table: Dict[int, float] = field(default_factory=dict)

    def calibrated_probability(self, market_price_cents: int) -> float:
        """
        Map market price (1-99 cents) to calibrated actual win rate.

        Parameters
        ----------
        market_price_cents : int
            Market price in cents (1-99).

        Returns
        -------
        float
            Calibrated probability of YES outcome.
        """
        market_price_cents = int(np.clip(market_price_cents, 1, 99))

        # Use empirical data if available
        if market_price_cents in self.empirical_table:
            return self.empirical_table[market_price_cents]

        # Parametric model: logit(actual) = gamma * logit(implied)
        implied = market_price_cents / 100.0
        calibrated = _inv_logit(self.gamma * _logit(implied))

        return float(np.clip(calibrated, 0.001, 0.999))

    def calibrated_edge(
        self,
        model_probability: float,
        market_price_cents: int,
    ) -> float:
        """
        Compute calibration-corrected edge.

        Standard edge: model_prob - market_price
        Calibrated edge: model_prob - calibrated_win_rate

        The calibrated edge is a more honest measure because it accounts
        for the market's systematic miscalibration at each price level.

        Parameters
        ----------
        model_probability : float
            Our model's predicted probability (0-1).
        market_price_cents : int
            Market price in cents (1-99).

        Returns
        -------
        float
            Calibrated edge (positive = model says more likely than data).
        """
        actual_win_rate = self.calibrated_probability(market_price_cents)
        return model_probability - actual_win_rate

    def yes_adjusted_edge(
        self,
        model_probability: float,
        market_price_cents: int,
    ) -> Tuple[float, float]:
        """
        Compute direction-adjusted edges for YES and NO sides.

        Accounts for the empirical finding that YES contracts are
        systematically overpriced (takers prefer buying YES).

        Parameters
        ----------
        model_probability : float
            Our model's predicted probability (0-1).
        market_price_cents : int
            Market price in cents (1-99).

        Returns
        -------
        tuple[float, float]
            (yes_edge, no_edge) — Positive values favor that direction.
            yes_edge = model_prob - (calibrated + yes_penalty)
            no_edge = (1 - model_prob) - (1 - calibrated + yes_penalty)
        """
        cal_prob = self.calibrated_probability(market_price_cents)

        # YES side: need to clear the calibrated rate PLUS the YES penalty
        yes_edge = model_probability - (cal_prob + self.yes_penalty)

        # NO side: benefits from the YES penalty (selling to biased buyers)
        no_edge = (1 - model_probability) - (1 - cal_prob - self.yes_penalty)

        return yes_edge, no_edge

    def load_empirical_data(self, price_win_rates: Dict[int, float]):
        """
        Load empirical calibration data from trade-level analysis.

        Parameters
        ----------
        price_win_rates : dict
            Mapping of price_cents (1-99) to actual win rate (0-1).
        """
        self.empirical_table = {
            int(k): float(v) for k, v in price_win_rates.items()
        }

    def mispricing_at(self, market_price_cents: int) -> float:
        """
        Compute mispricing at a given price level.

        Returns
        -------
        float
            Positive = market overprices YES (actual < implied).
            Negative = market underprices YES (actual > implied).
        """
        implied = market_price_cents / 100.0
        actual = self.calibrated_probability(market_price_cents)
        return implied - actual

    def as_table(self) -> List[Dict]:
        """Return full calibration table for inspection."""
        rows = []
        for cents in range(1, 100):
            implied = cents / 100.0
            actual = self.calibrated_probability(cents)
            rows.append({
                "price_cents": cents,
                "implied_probability": implied,
                "calibrated_probability": actual,
                "mispricing": implied - actual,
                "source": "empirical" if cents in self.empirical_table else "parametric",
            })
        return rows


# ──────────────────────────────────────────────────────────────────
# Module-level convenience functions using a shared default curve
# ──────────────────────────────────────────────────────────────────

_default_curve = KalshiCalibrationCurve()


def calibrated_probability(market_price_cents: int) -> float:
    """Map market price (cents) to calibrated win rate using default curve."""
    return _default_curve.calibrated_probability(market_price_cents)


def calibrated_edge(
    model_probability: float,
    market_price_cents: int,
) -> float:
    """Compute calibration-corrected edge using default curve."""
    return _default_curve.calibrated_edge(model_probability, market_price_cents)


def directional_bias_score(
    model_probability: float,
    market_price_cents: int,
) -> float:
    """
    Score from -1 (strong NO bias) to +1 (strong YES bias).

    Combines our model's informational edge with the structural
    microstructure bias. When both agree, the signal is strongest.

    Parameters
    ----------
    model_probability : float
        Our model's predicted probability (0-1).
    market_price_cents : int
        Market price in cents (1-99).

    Returns
    -------
    float
        Directional bias score. Positive favors YES, negative favors NO.
    """
    cal_prob = _default_curve.calibrated_probability(market_price_cents)
    implied = market_price_cents / 100.0

    # Informational edge: model vs calibrated reality
    info_edge = model_probability - cal_prob

    # Structural edge: calibrated reality vs market price
    # Positive means market overprices YES (reality is lower than price)
    structural_edge = cal_prob - implied

    # When both signals agree, compound them
    if (info_edge > 0 and structural_edge > 0) or (info_edge < 0 and structural_edge < 0):
        combined = info_edge + structural_edge
    else:
        # Conflicting signals: trust informational edge, discount by conflict
        combined = info_edge * 0.7

    return float(np.clip(combined, -1.0, 1.0))
