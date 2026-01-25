"""
Feature-aware earnings prediction model.

Combines multiple signal sources:
1. Historical mention rate (Beta-Binomial base)
2. Market price signal (Bayesian shrinkage toward market)
3. External features (earnings surprise, stock momentum)
4. Prior call patterns (sequential features)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, Dict, List

import numpy as np
import pandas as pd
from scipy import stats

from .base import EarningsModel
from .beta_binomial import BetaBinomialEarningsModel


class FeatureAwareEarningsModel(EarningsModel):
    """
    Feature-aware earnings mention prediction model.

    This model combines multiple signals:

    1. **Historical Base Rate**: Beta-Binomial posterior from past mentions
       - Provides baseline probability from historical data
       - Uses recency weighting (half-life parameter)

    2. **Market Price Signal**: Kalshi market consensus
       - Markets aggregate diverse information
       - Shrink predictions toward market with configurable weight

    3. **Feature Adjustments**: External signals that shift probability
       - Stock momentum (positive returns → bullish tone → more mentions)
       - Earnings surprise (beats → confident execs → more keyword use)
       - Prior mentions (persistence effect)
       - Volatility (high vol → more hedging language)

    Parameters
    ----------
    alpha_prior : float
        Beta distribution alpha prior (default: 1.0)
    beta_prior : float
        Beta distribution beta prior (default: 1.0)
    half_life : float
        Recency weighting half-life (default: 8.0)
    market_weight : float
        Weight for market price (0-1, default: 0.3)
        Final prob = (1-w)*model_prob + w*market_prob
    feature_coefficients : dict
        Coefficients for feature adjustments (learned or specified)
    use_features : bool
        Whether to use feature adjustments (default: True)
    """

    # Default feature coefficients (can be learned from data)
    DEFAULT_COEFFICIENTS = {
        # Stock features (positive return → more optimistic language)
        "stock_return_30d": 0.15,  # 10% return → +1.5% prob
        "stock_return_5d": 0.10,  # Short-term momentum
        "stock_volatility_30d": -0.05,  # High vol → less specific commitments

        # Earnings features (beats → more confident language)
        "eps_surprise_last": 0.002,  # 10% surprise → +2% prob
        "eps_beat_streak": 0.02,  # Each consecutive beat

        # Prior mention features (strong persistence)
        "mentioned_last_call": 0.15,  # Mentioned last time → likely again
        "mention_rate_4q": 0.20,  # Historical rate matters
        "mention_trend": 0.10,  # Increasing mentions

        # Market momentum
        "market_momentum": 0.30,  # Price moving up → information
    }

    def __init__(
        self,
        alpha_prior: float = 1.0,
        beta_prior: float = 1.0,
        half_life: Optional[float] = 8.0,
        market_weight: float = 0.3,
        feature_coefficients: Optional[Dict[str, float]] = None,
        use_features: bool = True,
        clip_adjustment: float = 0.15,  # Max feature adjustment
    ):
        self.alpha_prior = alpha_prior
        self.beta_prior = beta_prior
        self.half_life = half_life
        self.market_weight = market_weight
        self.use_features = use_features
        self.clip_adjustment = clip_adjustment

        # Feature coefficients
        self.feature_coefficients = feature_coefficients or self.DEFAULT_COEFFICIENTS.copy()

        # Internal Beta-Binomial model
        self._base_model = BetaBinomialEarningsModel(
            alpha_prior=alpha_prior,
            beta_prior=beta_prior,
            half_life=half_life,
        )

        # Fitted state
        self.fitted = False
        self._last_features: Optional[pd.DataFrame] = None

    def fit(
        self,
        features: pd.DataFrame,
        outcomes: pd.Series,
        market_features: Optional[pd.DataFrame] = None,
    ):
        """
        Fit model on historical data.

        Parameters
        ----------
        features : pd.DataFrame
            Word count features (can be ignored for pure Beta-Binomial)
        outcomes : pd.Series
            Binary outcomes (1=mentioned, 0=not mentioned)
        market_features : pd.DataFrame, optional
            Market features for calibration (not used in basic fit)
        """
        # Fit base Beta-Binomial model
        self._base_model.fit(features, outcomes)
        self.fitted = True

    def _calculate_feature_adjustment(
        self,
        market_features: Optional[Dict[str, float]] = None,
    ) -> float:
        """Calculate probability adjustment from features."""
        if not self.use_features or market_features is None:
            return 0.0

        adjustment = 0.0

        for feature_name, coefficient in self.feature_coefficients.items():
            if feature_name in market_features:
                value = market_features[feature_name]
                if value is not None and not np.isnan(value):
                    adjustment += coefficient * value

        # Clip to prevent extreme adjustments
        return np.clip(adjustment, -self.clip_adjustment, self.clip_adjustment)

    def predict(
        self,
        features: Optional[pd.DataFrame] = None,
        market_price: float = 0.5,
        market_features: Optional[Dict[str, float]] = None,
    ) -> pd.DataFrame:
        """
        Make prediction combining all signals.

        Parameters
        ----------
        features : pd.DataFrame, optional
            Current features (not used in basic prediction)
        market_price : float
            Kalshi market price (0-1)
        market_features : dict, optional
            Additional features for adjustment

        Returns
        -------
        pd.DataFrame
            Single row with: probability, lower_bound, upper_bound, uncertainty,
            base_prob, market_price, feature_adjustment
        """
        # Get base probability from Beta-Binomial
        base_pred = self._base_model.predict(features)
        base_prob = float(base_pred.iloc[0]["probability"])
        lower_bound = float(base_pred.iloc[0]["lower_bound"])
        upper_bound = float(base_pred.iloc[0]["upper_bound"])
        uncertainty = float(base_pred.iloc[0]["uncertainty"])

        # Calculate feature adjustment
        feature_adj = self._calculate_feature_adjustment(market_features)

        # Combine signals:
        # 1. Start with base probability
        # 2. Apply feature adjustment
        # 3. Shrink toward market price

        # Apply feature adjustment to base
        adjusted_prob = base_prob + feature_adj

        # Shrink toward market (Bayesian compromise)
        # Higher market_weight = trust market more
        final_prob = (1 - self.market_weight) * adjusted_prob + self.market_weight * market_price

        # Ensure valid probability
        final_prob = np.clip(final_prob, 0.01, 0.99)

        # Adjust bounds for feature adjustment
        adj_lower = np.clip(lower_bound + feature_adj, 0.01, final_prob)
        adj_upper = np.clip(upper_bound + feature_adj, final_prob, 0.99)

        return pd.DataFrame([{
            "probability": final_prob,
            "lower_bound": adj_lower,
            "upper_bound": adj_upper,
            "uncertainty": uncertainty,
            "base_probability": base_prob,
            "market_price": market_price,
            "feature_adjustment": feature_adj,
        }])

    def get_signal_decomposition(
        self,
        features: Optional[pd.DataFrame] = None,
        market_price: float = 0.5,
        market_features: Optional[Dict[str, float]] = None,
    ) -> Dict[str, float]:
        """
        Decompose prediction into component signals.

        Useful for understanding what drives the prediction.
        """
        pred = self.predict(features, market_price, market_features)

        base_prob = float(pred.iloc[0]["base_probability"])
        final_prob = float(pred.iloc[0]["probability"])
        feature_adj = float(pred.iloc[0]["feature_adjustment"])

        # Calculate each component's contribution
        market_contribution = self.market_weight * (market_price - base_prob)
        feature_contribution = (1 - self.market_weight) * feature_adj

        return {
            "final_probability": final_prob,
            "base_probability": base_prob,
            "market_price": market_price,
            "market_weight": self.market_weight,
            "market_contribution": market_contribution,
            "feature_adjustment": feature_adj,
            "feature_contribution": feature_contribution,
        }

    def save(self, filepath: str):
        """Save model parameters to JSON."""
        data = {
            "model_type": "FeatureAwareEarningsModel",
            "alpha_prior": self.alpha_prior,
            "beta_prior": self.beta_prior,
            "half_life": self.half_life,
            "market_weight": self.market_weight,
            "use_features": self.use_features,
            "clip_adjustment": self.clip_adjustment,
            "feature_coefficients": self.feature_coefficients,
            "base_model": {
                "alpha_post": self._base_model.alpha_post,
                "beta_post": self._base_model.beta_post,
                "n_observations": self._base_model.n_observations,
            },
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
        self.market_weight = data["market_weight"]
        self.use_features = data["use_features"]
        self.clip_adjustment = data["clip_adjustment"]
        self.feature_coefficients = data["feature_coefficients"]

        # Restore base model
        self._base_model.alpha_prior = self.alpha_prior
        self._base_model.beta_prior = self.beta_prior
        self._base_model.half_life = self.half_life
        self._base_model.alpha_post = data["base_model"]["alpha_post"]
        self._base_model.beta_post = data["base_model"]["beta_post"]
        self._base_model.n_observations = data["base_model"]["n_observations"]

        self.fitted = True


class AdaptiveFeatureModel(FeatureAwareEarningsModel):
    """
    Feature-aware model that learns coefficients from data.

    Uses gradient-free optimization to find feature coefficients
    that minimize prediction error on historical data.
    """

    def __init__(
        self,
        alpha_prior: float = 1.0,
        beta_prior: float = 1.0,
        half_life: Optional[float] = 8.0,
        market_weight: float = 0.3,
        use_features: bool = True,
        learning_rate: float = 0.01,
        regularization: float = 0.1,
    ):
        super().__init__(
            alpha_prior=alpha_prior,
            beta_prior=beta_prior,
            half_life=half_life,
            market_weight=market_weight,
            feature_coefficients=None,  # Will be learned
            use_features=use_features,
        )

        self.learning_rate = learning_rate
        self.regularization = regularization

        # Initialize coefficients to small values
        self.feature_coefficients = {
            k: 0.01 for k in self.DEFAULT_COEFFICIENTS
        }

    def fit_with_features(
        self,
        features: pd.DataFrame,
        outcomes: pd.Series,
        market_features_df: pd.DataFrame,
        market_prices: pd.Series,
        n_iterations: int = 100,
    ):
        """
        Fit model including learning feature coefficients.

        Uses simple coordinate descent to minimize Brier score.
        """
        # First fit base model
        self.fit(features, outcomes)

        # Learn feature coefficients
        feature_names = list(self.feature_coefficients.keys())
        available_features = [f for f in feature_names if f in market_features_df.columns]

        if not available_features:
            return

        for iteration in range(n_iterations):
            total_error = 0.0

            for idx in outcomes.index:
                if idx not in market_features_df.index:
                    continue

                actual = outcomes.loc[idx]
                market_price = market_prices.loc[idx] if idx in market_prices.index else 0.5

                # Get features for this observation
                obs_features = market_features_df.loc[idx].to_dict()

                # Make prediction
                pred = self.predict(None, market_price, obs_features)
                predicted = float(pred.iloc[0]["probability"])

                # Calculate error gradient
                error = predicted - actual
                total_error += error ** 2

                # Update coefficients (gradient descent)
                for feature_name in available_features:
                    if feature_name in obs_features:
                        value = obs_features[feature_name]
                        if value is not None and not np.isnan(value):
                            gradient = error * value * (1 - self.market_weight)
                            gradient += self.regularization * self.feature_coefficients[feature_name]

                            self.feature_coefficients[feature_name] -= self.learning_rate * gradient

            # Clip coefficients to reasonable range
            for k in self.feature_coefficients:
                self.feature_coefficients[k] = np.clip(
                    self.feature_coefficients[k], -0.5, 0.5
                )
