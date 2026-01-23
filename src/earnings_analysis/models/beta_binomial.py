"""
Beta-Binomial model for earnings prediction.

Adapted from FOMC Beta-Binomial model for earnings calls.
Uses Bayesian approach with Beta prior and binomial likelihood.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats

from .base import EarningsModel


class BetaBinomialEarningsModel(EarningsModel):
    """
    Beta-Binomial Bayesian model for earnings prediction.

    Parameters
    ----------
    alpha_prior : float
        Alpha parameter for Beta prior (default: 1.0 = uniform prior)
    beta_prior : float
        Beta parameter for Beta prior (default: 1.0 = uniform prior)
    half_life : Optional[float]
        Half-life for exponential weighting (in number of events)
        If None, uses uniform weighting
    """

    def __init__(
        self,
        alpha_prior: float = 1.0,
        beta_prior: float = 1.0,
        half_life: Optional[float] = None,
    ):
        self.alpha_prior = alpha_prior
        self.beta_prior = beta_prior
        self.half_life = half_life

        # Fitted parameters
        self.alpha_post = None
        self.beta_post = None
        self.n_observations = 0

    def fit(self, features: pd.DataFrame, outcomes: pd.Series):
        """
        Fit Beta-Binomial model.

        Parameters
        ----------
        features : pd.DataFrame
            Feature matrix (not used in simple model, but kept for interface)
        outcomes : pd.Series
            Binary outcomes (1 or 0)
        """
        outcomes = outcomes.values

        # Calculate weights
        n = len(outcomes)
        if self.half_life is not None and n > 0:
            # Exponential weighting
            decay_rate = np.log(2) / self.half_life
            ages = np.arange(n, 0, -1)  # Most recent = 1, oldest = n
            weights = np.exp(-decay_rate * ages)
            weights = weights / weights.sum() * n  # Normalize to sum to n
        else:
            # Uniform weighting
            weights = np.ones(n)

        # Update Beta parameters
        successes = np.sum(outcomes * weights)
        failures = np.sum((1 - outcomes) * weights)

        self.alpha_post = self.alpha_prior + successes
        self.beta_post = self.beta_prior + failures
        self.n_observations = n

    def predict(self, features: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Make prediction using posterior distribution.

        Returns
        -------
        pd.DataFrame
            Single row with: probability, lower_bound, upper_bound, uncertainty
        """
        if self.alpha_post is None or self.beta_post is None:
            # Not fitted yet, return prior
            alpha = self.alpha_prior
            beta = self.beta_prior
        else:
            alpha = self.alpha_post
            beta = self.beta_post

        # Posterior mean
        probability = alpha / (alpha + beta)

        # 95% credible interval
        lower_bound = stats.beta.ppf(0.025, alpha, beta)
        upper_bound = stats.beta.ppf(0.975, alpha, beta)

        # Uncertainty (standard deviation)
        uncertainty = np.sqrt(alpha * beta / ((alpha + beta) ** 2 * (alpha + beta + 1)))

        return pd.DataFrame([{
            "probability": probability,
            "lower_bound": lower_bound,
            "upper_bound": upper_bound,
            "uncertainty": uncertainty,
        }])

    def save(self, filepath: str):
        """Save model parameters to JSON."""
        data = {
            "model_type": "BetaBinomialEarningsModel",
            "alpha_prior": self.alpha_prior,
            "beta_prior": self.beta_prior,
            "half_life": self.half_life,
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
        self.alpha_post = data["alpha_post"]
        self.beta_post = data["beta_post"]
        self.n_observations = data["n_observations"]
