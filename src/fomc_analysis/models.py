"""
Mention probability models with uncertainty estimation.

This module provides baseline models for estimating the probability that
a contract will be mentioned in a future press conference. All models
output both point estimates and uncertainty intervals.

Models:
- EWMAModel: Exponentially weighted moving average with bootstrap uncertainty
- BetaBinomialModel: Bayesian Beta-Binomial with credible intervals
- LogisticModel: Logistic regression with prediction intervals
"""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional, Dict, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression


@dataclass
class PredictionResult:
    """
    Prediction result with uncertainty.

    Attributes
    ----------
    probability : float
        Point estimate of mention probability.
    lower_bound : float
        Lower bound of credible/confidence interval.
    upper_bound : float
        Upper bound of credible/confidence interval.
    uncertainty : float
        Measure of uncertainty (std dev or interval width).
    """
    probability: float
    lower_bound: float
    upper_bound: float
    uncertainty: float


class BaseModel:
    """Base class for mention probability models."""

    def fit(self, events: pd.DataFrame) -> None:
        """
        Fit the model on binary event data.

        Parameters
        ----------
        events : pd.DataFrame
            Binary event matrix (rows=dates, cols=contracts).
        """
        raise NotImplementedError

    def predict(self, n_future: int = 1) -> pd.DataFrame:
        """
        Predict probabilities for next n_future events.

        Parameters
        ----------
        n_future : int, default=1
            Number of future events to predict.

        Returns
        -------
        pd.DataFrame
            DataFrame with columns: contract, probability, lower, upper, uncertainty.
        """
        raise NotImplementedError

    def save(self, path: Path) -> None:
        """Save model to disk."""
        raise NotImplementedError

    @classmethod
    def load(cls, path: Path) -> BaseModel:
        """Load model from disk."""
        raise NotImplementedError


class EWMAModel(BaseModel):
    """
    Exponentially weighted moving average model.

    This model computes EWMA probabilities and uses bootstrap resampling
    to estimate uncertainty.

    Parameters
    ----------
    alpha : float, default=0.5
        Smoothing parameter (higher = more weight on recent events).
    bootstrap_samples : int, default=100
        Number of bootstrap samples for uncertainty estimation.
    """

    def __init__(self, alpha: float = 0.5, bootstrap_samples: int = 100):
        self.alpha = alpha
        self.bootstrap_samples = bootstrap_samples
        self.fitted = False

    def fit(self, events: pd.DataFrame) -> None:
        """Fit EWMA model on event history."""
        self.events = events.copy()
        self.contracts = list(events.columns)
        self.fitted = True

    def predict(self, n_future: int = 1) -> pd.DataFrame:
        """
        Predict next event probability with bootstrap uncertainty.

        For EWMA, we use the latest EWMA value as the point estimate
        and bootstrap past residuals to estimate uncertainty.
        """
        if not self.fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")

        results = []

        for contract in self.contracts:
            series = self.events[contract].values

            # Compute EWMA
            ewma = self._compute_ewma(series)

            # Latest EWMA value is the prediction
            prob = ewma[-1]

            # Bootstrap uncertainty
            lower, upper = self._bootstrap_uncertainty(series, prob)

            results.append({
                "contract": contract,
                "probability": prob,
                "lower_bound": lower,
                "upper_bound": upper,
                "uncertainty": (upper - lower) / 2,
            })

        return pd.DataFrame(results)

    def _compute_ewma(self, series: np.ndarray) -> np.ndarray:
        """Compute EWMA for a time series."""
        ewma = np.zeros(len(series))
        ewma[0] = 0.5  # Prior

        for i in range(1, len(series)):
            ewma[i] = self.alpha * series[i-1] + (1 - self.alpha) * ewma[i-1]

        return ewma

    def _bootstrap_uncertainty(
        self,
        series: np.ndarray,
        point_estimate: float,
        confidence: float = 0.9,
    ) -> Tuple[float, float]:
        """
        Bootstrap confidence interval for EWMA prediction.

        We resample residuals and recompute EWMA to get distribution.
        """
        if len(series) < 5:
            # Not enough data, use wide interval
            return max(0, point_estimate - 0.3), min(1, point_estimate + 0.3)

        # Compute residuals from EWMA fit
        ewma = self._compute_ewma(series)
        residuals = series - ewma

        # Bootstrap
        predictions = []
        for _ in range(self.bootstrap_samples):
            # Resample residuals
            resampled_residuals = np.random.choice(
                residuals, size=len(residuals), replace=True
            )
            resampled_series = ewma + resampled_residuals

            # Clip to [0, 1]
            resampled_series = np.clip(resampled_series, 0, 1)

            # Recompute EWMA on resampled data
            resampled_ewma = self._compute_ewma(resampled_series)
            predictions.append(resampled_ewma[-1])

        # Compute percentiles
        lower = np.percentile(predictions, (1 - confidence) / 2 * 100)
        upper = np.percentile(predictions, (1 + confidence) / 2 * 100)

        return float(lower), float(upper)

    def save(self, path: Path) -> None:
        """Save model parameters."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "model_type": "EWMA",
            "alpha": self.alpha,
            "bootstrap_samples": self.bootstrap_samples,
            "contracts": self.contracts,
            "events": self.events.to_dict(orient="list"),
        }

        path.write_text(json.dumps(data, indent=2))

    @classmethod
    def load(cls, path: Path) -> EWMAModel:
        """Load model from file."""
        data = json.loads(Path(path).read_text())
        model = cls(
            alpha=data["alpha"],
            bootstrap_samples=data["bootstrap_samples"],
        )
        model.events = pd.DataFrame(data["events"])
        model.contracts = data["contracts"]
        model.fitted = True
        return model


class BetaBinomialModel(BaseModel):
    """
    Bayesian Beta-Binomial model with credible intervals.

    This model uses a Beta prior and updates it with observed events.
    Credible intervals come naturally from the posterior distribution.

    Parameters
    ----------
    alpha_prior : float, default=1.0
        Alpha parameter of Beta prior.
    beta_prior : float, default=1.0
        Beta parameter of Beta prior.
    half_life : Optional[int], default=None
        Half-life for exponential decay (None = no decay).
    """

    def __init__(
        self,
        alpha_prior: float = 1.0,
        beta_prior: float = 1.0,
        half_life: Optional[int] = None,
    ):
        self.alpha_prior = alpha_prior
        self.beta_prior = beta_prior
        self.half_life = half_life
        self.fitted = False

    def fit(self, events: pd.DataFrame) -> None:
        """Fit Beta-Binomial model."""
        self.events = events.copy()
        self.contracts = list(events.columns)
        self.fitted = True

    def predict(self, n_future: int = 1, credible_mass: float = 0.9) -> pd.DataFrame:
        """
        Predict with Bayesian credible intervals.

        The posterior after observing k successes in n trials is
        Beta(alpha + k, beta + n - k). We report the mean and
        credible interval.
        """
        if not self.fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")

        results = []

        # Compute decay weights if half_life is set
        n_obs = len(self.events)
        if self.half_life is not None and self.half_life > 0:
            decay_factor = 0.5 ** (1.0 / self.half_life)
            weights = np.array([decay_factor ** (n_obs - i - 1) for i in range(n_obs)])
        else:
            weights = np.ones(n_obs)

        for contract in self.contracts:
            series = self.events[contract].to_numpy(dtype=float)

            # Ignore meetings where the contract did not exist (NaN entries)
            valid_mask = np.isfinite(series)
            if valid_mask.any():
                valid_series = series[valid_mask]
                valid_weights = weights[valid_mask]

                # Compute weighted successes and trials using only observed data
                successes = float(np.sum(valid_series * valid_weights))
                trials = float(np.sum(valid_weights))
            else:
                # No historical observations; fall back to the prior
                successes = 0.0
                trials = 0.0

            # Posterior parameters
            alpha_post = self.alpha_prior + successes
            beta_post = self.beta_prior + (trials - successes)

            # Mean of Beta distribution
            prob = alpha_post / (alpha_post + beta_post)

            # Credible interval
            lower_percentile = (1 - credible_mass) / 2
            upper_percentile = (1 + credible_mass) / 2

            lower = stats.beta.ppf(lower_percentile, alpha_post, beta_post)
            upper = stats.beta.ppf(upper_percentile, alpha_post, beta_post)

            # Uncertainty (standard deviation of posterior)
            uncertainty = np.sqrt(
                (alpha_post * beta_post) /
                ((alpha_post + beta_post) ** 2 * (alpha_post + beta_post + 1))
            )

            results.append({
                "contract": contract,
                "probability": float(prob),
                "lower_bound": float(lower),
                "upper_bound": float(upper),
                "uncertainty": float(uncertainty),
            })

        return pd.DataFrame(results)

    def save(self, path: Path) -> None:
        """Save model."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "model_type": "BetaBinomial",
            "alpha_prior": self.alpha_prior,
            "beta_prior": self.beta_prior,
            "half_life": self.half_life,
            "contracts": self.contracts,
            "events": self.events.to_dict(orient="list"),
        }

        path.write_text(json.dumps(data, indent=2))

    @classmethod
    def load(cls, path: Path) -> BetaBinomialModel:
        """Load model from file."""
        data = json.loads(Path(path).read_text())
        model = cls(
            alpha_prior=data["alpha_prior"],
            beta_prior=data["beta_prior"],
            half_life=data.get("half_life"),
        )
        model.events = pd.DataFrame(data["events"])
        model.contracts = data["contracts"]
        model.fitted = True
        return model
