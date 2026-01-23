"""
Sentiment-based model for earnings prediction.

Uses sentiment features to predict price movements.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from .base import EarningsModel


class SentimentBasedModel(EarningsModel):
    """
    Logistic regression model using sentiment features.

    Parameters
    ----------
    feature_columns : list
        Columns to use as features (e.g., sentiment scores, keyword counts)
    """

    def __init__(self, feature_columns: list = None):
        self.feature_columns = feature_columns or [
            "sentiment_positive_count",
            "sentiment_negative_count",
            "guidance_count",
            "ceo_word_count",
            "cfo_word_count",
        ]

        self.model = LogisticRegression(random_state=42)
        self.is_fitted = False

    def fit(self, features: pd.DataFrame, outcomes: pd.Series):
        """
        Fit logistic regression model.

        Parameters
        ----------
        features : pd.DataFrame
            Feature matrix
        outcomes : pd.Series
            Binary outcomes (1 or 0)
        """
        # Select feature columns
        X = features[self.feature_columns].fillna(0).values
        y = outcomes.values

        # Fit model
        self.model.fit(X, y)
        self.is_fitted = True

    def predict(self, features: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Make predictions using logistic regression.

        Parameters
        ----------
        features : Optional[pd.DataFrame]
            Features for prediction

        Returns
        -------
        pd.DataFrame
            Predictions with probability and confidence intervals
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted yet")

        if features is None:
            raise ValueError("Features required for prediction")

        X = features[self.feature_columns].fillna(0).values

        # Get probabilities
        probabilities = self.model.predict_proba(X)[:, 1]

        # Simple confidence intervals (could be improved with bootstrapping)
        lower_bounds = np.maximum(probabilities - 0.15, 0)
        upper_bounds = np.minimum(probabilities + 0.15, 1)

        return pd.DataFrame({
            "probability": probabilities,
            "lower_bound": lower_bounds,
            "upper_bound": upper_bounds,
            "uncertainty": 0.15,  # Fixed uncertainty for now
        })

    def save(self, filepath: str):
        """Save model to file."""
        import pickle

        data = {
            "feature_columns": self.feature_columns,
            "model": self.model,
            "is_fitted": self.is_fitted,
        }

        Path(filepath).parent.mkdir(parents=True, exist_ok=True)

        with open(filepath, "wb") as f:
            pickle.dump(data, f)

    def load(self, filepath: str):
        """Load model from file."""
        import pickle

        with open(filepath, "rb") as f:
            data = pickle.load(f)

        self.feature_columns = data["feature_columns"]
        self.model = data["model"]
        self.is_fitted = data["is_fitted"]
