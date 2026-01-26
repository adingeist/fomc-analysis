"""
Market-Adjusted Beta-Binomial model for earnings prediction.

This model improves upon the basic Beta-Binomial by:
1. Shrinking predictions towards market prices when data is limited
2. Providing confidence-adjusted edge calculations
3. Using Kelly criterion for position sizing recommendations
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, Dict, Any

import numpy as np
import pandas as pd
from scipy import stats

from .base import EarningsModel


class MarketAdjustedModel(EarningsModel):
    """
    Market-Adjusted Beta-Binomial model.

    Key improvement: When historical data is limited, the model shrinks
    its predictions towards the market price rather than using uninformed
    category priors.

    Parameters
    ----------
    alpha_prior : float
        Alpha parameter for Beta prior (default: 1.0)
    beta_prior : float
        Beta parameter for Beta prior (default: 1.0)
    half_life : Optional[float]
        Half-life for exponential weighting (default: 4.0)
    shrinkage_samples : int
        Number of samples at which shrinkage towards market is 50% (default: 3)
    min_samples_to_trade : int
        Minimum samples required to generate a trade signal (default: 2)
    confidence_level : float
        Confidence level for credible intervals (default: 0.80)
    """

    def __init__(
        self,
        alpha_prior: float = 1.0,
        beta_prior: float = 1.0,
        half_life: Optional[float] = 4.0,
        shrinkage_samples: int = 3,
        min_samples_to_trade: int = 2,
        confidence_level: float = 0.80,
    ):
        self.alpha_prior = alpha_prior
        self.beta_prior = beta_prior
        self.half_life = half_life
        self.shrinkage_samples = shrinkage_samples
        self.min_samples_to_trade = min_samples_to_trade
        self.confidence_level = confidence_level

        # Fitted parameters
        self.alpha_post = None
        self.beta_post = None
        self.n_observations = 0
        self.raw_probability = None

    def fit(self, features: pd.DataFrame, outcomes: pd.Series):
        """
        Fit Beta-Binomial model with recency weighting.
        """
        outcomes = outcomes.values
        n = len(outcomes)

        if self.half_life is not None and n > 0:
            decay_rate = np.log(2) / self.half_life
            ages = np.arange(n, 0, -1)
            weights = np.exp(-decay_rate * ages)
            weights = weights / weights.sum() * n
        else:
            weights = np.ones(n)

        successes = np.sum(outcomes * weights)
        failures = np.sum((1 - outcomes) * weights)

        self.alpha_post = self.alpha_prior + successes
        self.beta_post = self.beta_prior + failures
        self.n_observations = n
        self.raw_probability = self.alpha_post / (self.alpha_post + self.beta_post)

    def predict(
        self,
        features: Optional[pd.DataFrame] = None,
        market_price: Optional[float] = None,
    ) -> pd.DataFrame:
        """
        Make prediction with market-adjusted shrinkage.

        Parameters
        ----------
        features : Optional[pd.DataFrame]
            Features (not used in this model)
        market_price : Optional[float]
            Current market price (0-1). If provided, prediction is shrunk
            towards market price based on sample size.

        Returns
        -------
        pd.DataFrame
            Prediction with probability, bounds, confidence, and trade signal
        """
        if self.alpha_post is None or self.beta_post is None:
            alpha = self.alpha_prior
            beta = self.beta_prior
            n = 0
        else:
            alpha = self.alpha_post
            beta = self.beta_post
            n = self.n_observations

        # Raw model probability
        raw_prob = alpha / (alpha + beta)

        # Apply shrinkage towards market if market price provided
        if market_price is not None and n < self.shrinkage_samples * 3:
            # Shrinkage factor: 0 when n=0 (use market), 1 when n is large (use model)
            shrinkage = n / (n + self.shrinkage_samples)
            probability = shrinkage * raw_prob + (1 - shrinkage) * market_price
        else:
            probability = raw_prob

        # Credible interval
        alpha_ci = (1 - self.confidence_level) / 2
        lower_bound = stats.beta.ppf(alpha_ci, alpha, beta)
        upper_bound = stats.beta.ppf(1 - alpha_ci, alpha, beta)

        # Uncertainty
        uncertainty = np.sqrt(alpha * beta / ((alpha + beta) ** 2 * (alpha + beta + 1)))

        # Calculate edge and confidence-adjusted edge
        edge = None
        adjusted_edge = None
        trade_signal = "HOLD"
        kelly_fraction = 0.0

        if market_price is not None:
            edge = probability - market_price

            # Confidence adjustment: reduce edge based on uncertainty
            confidence_factor = min(1.0, n / (self.min_samples_to_trade * 2))
            adjusted_edge = edge * confidence_factor

            # Trade signal
            if n >= self.min_samples_to_trade:
                if adjusted_edge > 0.10:
                    trade_signal = "BUY_YES"
                elif adjusted_edge < -0.10:
                    trade_signal = "BUY_NO"

            # Kelly criterion for position sizing
            if trade_signal != "HOLD":
                if trade_signal == "BUY_YES":
                    # Betting on YES: p = probability, b = (1-market_price)/market_price
                    p = probability
                    b = (1 - market_price) / market_price if market_price > 0 else 0
                else:
                    # Betting on NO: p = 1-probability, b = market_price/(1-market_price)
                    p = 1 - probability
                    b = market_price / (1 - market_price) if market_price < 1 else 0

                if b > 0:
                    kelly_fraction = max(0, (p * b - (1 - p)) / b)
                    # Apply half-Kelly and confidence adjustment
                    kelly_fraction = kelly_fraction * 0.5 * confidence_factor

        return pd.DataFrame([{
            "probability": probability,
            "raw_probability": raw_prob,
            "lower_bound": lower_bound,
            "upper_bound": upper_bound,
            "uncertainty": uncertainty,
            "n_samples": n,
            "market_price": market_price,
            "edge": edge,
            "adjusted_edge": adjusted_edge,
            "trade_signal": trade_signal,
            "kelly_fraction": kelly_fraction,
            "confidence": confidence_factor if market_price is not None else None,
        }])

    def get_trade_recommendation(
        self,
        market_price: float,
        position_size_pct: float = 0.03,
        max_position_pct: float = 0.10,
    ) -> Dict[str, Any]:
        """
        Get detailed trade recommendation.

        Parameters
        ----------
        market_price : float
            Current market price (0-1)
        position_size_pct : float
            Base position size as fraction of capital (default: 3%)
        max_position_pct : float
            Maximum position size (default: 10%)

        Returns
        -------
        Dict with trade recommendation details
        """
        pred = self.predict(market_price=market_price)
        row = pred.iloc[0]

        # Scale position by Kelly and confidence
        kelly = row['kelly_fraction']
        recommended_position = min(
            position_size_pct * (1 + kelly * 2),  # Scale up based on Kelly
            max_position_pct
        )

        return {
            "signal": row['trade_signal'],
            "probability": row['probability'],
            "market_price": market_price,
            "edge": row['edge'],
            "adjusted_edge": row['adjusted_edge'],
            "confidence": row['confidence'],
            "recommended_position_pct": recommended_position if row['trade_signal'] != "HOLD" else 0,
            "kelly_fraction": kelly,
            "n_samples": row['n_samples'],
            "lower_bound": row['lower_bound'],
            "upper_bound": row['upper_bound'],
        }

    def save(self, filepath: str):
        """Save model parameters to JSON."""
        data = {
            "model_type": "MarketAdjustedModel",
            "alpha_prior": self.alpha_prior,
            "beta_prior": self.beta_prior,
            "half_life": self.half_life,
            "shrinkage_samples": self.shrinkage_samples,
            "min_samples_to_trade": self.min_samples_to_trade,
            "confidence_level": self.confidence_level,
            "alpha_post": self.alpha_post,
            "beta_post": self.beta_post,
            "n_observations": self.n_observations,
        }

        Path(filepath).parent.mkdir(parents=True, exist_ok=True)

        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)

    def load(self, filepath: str):
        """Load model parameters from JSON."""
        with open(filepath, "r") as f:
            data = json.load(f)

        self.alpha_prior = data["alpha_prior"]
        self.beta_prior = data["beta_prior"]
        self.half_life = data["half_life"]
        self.shrinkage_samples = data.get("shrinkage_samples", 3)
        self.min_samples_to_trade = data.get("min_samples_to_trade", 2)
        self.confidence_level = data.get("confidence_level", 0.80)
        self.alpha_post = data["alpha_post"]
        self.beta_post = data["beta_post"]
        self.n_observations = data["n_observations"]
