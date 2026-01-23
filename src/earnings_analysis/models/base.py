"""
Base model interface for earnings prediction.

All earnings models should implement this interface.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

import pandas as pd


class EarningsModel(ABC):
    """
    Base class for earnings prediction models.

    Models predict binary outcomes (e.g., price up/down, beat/miss)
    with probability estimates and confidence intervals.
    """

    @abstractmethod
    def fit(self, features: pd.DataFrame, outcomes: pd.Series):
        """
        Fit the model on historical data.

        Parameters
        ----------
        features : pd.DataFrame
            Feature matrix (one row per earnings call)
        outcomes : pd.Series
            Binary outcomes (1 or 0)
        """
        pass

    @abstractmethod
    def predict(self, features: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Make predictions.

        Parameters
        ----------
        features : Optional[pd.DataFrame]
            Features for prediction. If None, predicts for next period.

        Returns
        -------
        pd.DataFrame
            Predictions with columns: probability, lower_bound, upper_bound, uncertainty
        """
        pass

    @abstractmethod
    def save(self, filepath: str):
        """Save model to file."""
        pass

    @abstractmethod
    def load(self, filepath: str):
        """Load model from file."""
        pass
