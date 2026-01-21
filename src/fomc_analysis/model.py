"""
model
=====

This module defines a family of simple estimators for the probability
that a given contract will be mentioned in a future press conference.
The base class :class:`EstimateModel` defines the interface; specific
models implement the ``fit`` and ``predict_proba`` methods.  Models
operate on binary event matrices (1 if a contract was mentioned at
least once, 0 otherwise) or on more detailed features extracted from
the transcripts.

Available models:

* :class:`EwmaModel` – applies exponential smoothing to past events.
* :class:`BetaBinomialModel` – updates a Beta–Binomial posterior with
  optional exponential decay.
* :class:`LogisticRegressionModel` – uses scikit‑learn's logistic
  regression to learn from a feature matrix and predict mention
  probabilities.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict, Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from .feature_extraction import ewma_probabilities, beta_binomial_estimator


class EstimateModel:
    """Base class for mention probability estimators.

    Subclasses should override :meth:`fit` and :meth:`predict_proba`.
    The ``fit`` method prepares the model using historical data; the
    ``predict_proba`` method produces probabilities for the input data.
    """

    def fit(self, events: pd.DataFrame, **kwargs: Any) -> None:
        """Fit the model on a DataFrame of binary events.

        Parameters
        ----------
        events: pandas.DataFrame
            Binary matrix where rows correspond to time periods and
            columns correspond to contracts.  A value of 1 indicates
            that the contract was mentioned; 0 otherwise.

        Raises
        ------
        NotImplementedError
            If the subclass does not implement this method.
        """
        raise NotImplementedError

    def predict_proba(self, events: pd.DataFrame, **kwargs: Any) -> pd.DataFrame:
        """Predict mention probabilities for each row in ``events``.

        Parameters
        ----------
        events: pandas.DataFrame
            Binary event matrix for which to estimate probabilities.

        Returns
        -------
        pandas.DataFrame
            A DataFrame of probabilities with the same shape and
            index/columns as ``events``.
        """
        raise NotImplementedError


@dataclass
class EwmaModel(EstimateModel):
    """Exponentially weighted moving average model.

    Parameters
    ----------
    alpha: float, default 0.5
        Smoothing parameter.  Higher values weight recent observations
        more heavily.
    init: float, default 0.5
        Initial probability used for the first period.  Must lie
        between 0 and 1.
    """

    alpha: float = 0.5
    init: float = 0.5

    def fit(self, events: pd.DataFrame, **kwargs: Any) -> None:
        # For EWMA, no explicit fitting is required since the
        # probabilities are computed directly from the events.  The
        # model stores the training events in case they are needed for
        # evaluation or baseline probabilities.
        self.train_events = events.copy()
        self.train_index = events.index
        return None

    def predict_proba(self, events: pd.DataFrame, **kwargs: Any) -> pd.DataFrame:
        # Compute EWMA probabilities on the concatenation of training
        # and prediction data.  We drop the training part of the
        # result to return only the probabilities for the new data.
        if not hasattr(self, "train_events"):
            raise RuntimeError("EwmaModel has not been fitted.")
        combined = pd.concat([self.train_events, events], axis=0)
        probs = ewma_probabilities(combined, alpha=self.alpha)
        # Drop training rows
        return probs.loc[events.index].astype(float)


@dataclass
class BetaBinomialModel(EstimateModel):
    """Beta–Binomial estimator with optional exponential decay.

    Parameters
    ----------
    alpha_prior: float, default 1.0
        Alpha parameter of the Beta prior.
    beta_prior: float, default 1.0
        Beta parameter of the Beta prior.
    half_life: Optional[int], default None
        Number of observations corresponding to a half‑life for
        exponential decay.  If None, no decay is applied.
    """

    alpha_prior: float = 1.0
    beta_prior: float = 1.0
    half_life: Optional[int] = None

    def fit(self, events: pd.DataFrame, **kwargs: Any) -> None:
        self.train_events = events.copy()
        return None

    def predict_proba(self, events: pd.DataFrame, **kwargs: Any) -> pd.DataFrame:
        if not hasattr(self, "train_events"):
            raise RuntimeError("BetaBinomialModel has not been fitted.")
        combined = pd.concat([self.train_events, events], axis=0)
        probs = beta_binomial_estimator(
            combined,
            alpha_prior=self.alpha_prior,
            beta_prior=self.beta_prior,
            half_life=self.half_life,
        )
        return probs.loc[events.index].astype(float)


@dataclass
class LogisticRegressionModel(EstimateModel):
    """Multivariate logistic regression model.

    This model treats past binary events or other numerical features as
    predictors for future mention probability.  For each contract it
    fits a separate logistic regression classifier.  You can supply
    additional features via the ``features`` argument of
    :meth:`predict_proba`.

    Parameters
    ----------
    max_iter: int, default 1000
        Maximum number of iterations for the solver.
    C: float, default 1.0
        Inverse of regularisation strength for scikit‑learn's
        LogisticRegression.
    solver: str, default "lbfgs"
        Optimisation algorithm used by scikit‑learn.  See
        scikit‑learn's documentation for choices.
    """

    max_iter: int = 1000
    C: float = 1.0
    solver: str = "lbfgs"

    def fit(self, events: pd.DataFrame, features: Optional[pd.DataFrame] = None) -> None:
        """Fit the logistic regression models.

        Parameters
        ----------
        events: pandas.DataFrame
            Binary event matrix.  Each column will have its own
            logistic regression model trained to predict that column.
        features: pandas.DataFrame, optional
            Additional features for each row.  If provided, it should
            have the same index as ``events``.  If None, uses the
            lagged event matrix as features (i.e. each column's past
            values predict its own future value).
        """
        self.contracts_ = list(events.columns)
        # Use lagged events as default features
        if features is None:
            # Lag events by one period (drop the first row)
            lagged = events.shift(1).fillna(0)
            X = lagged
        else:
            X = features
        self.models_: Dict[str, LogisticRegression] = {}
        for contract in self.contracts_:
            y = events[contract]
            # Fit logistic regression for this contract
            model = LogisticRegression(max_iter=self.max_iter, C=self.C, solver=self.solver)
            model.fit(X.values, y.values)
            self.models_[contract] = model
        self.train_features_ = X
        return None

    def predict_proba(self, events: pd.DataFrame, features: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """Predict mention probabilities using the fitted models.

        Parameters
        ----------
        events: pandas.DataFrame
            A DataFrame with the same columns as used during fitting.
        features: pandas.DataFrame, optional
            Additional features aligned with the rows of ``events``.
            If None, uses the lagged events from the concatenation of
            training and prediction events.

        Returns
        -------
        pandas.DataFrame
            A DataFrame of probabilities with the same index and
            columns as ``events``.
        """
        if not hasattr(self, "models_"):
            raise RuntimeError("LogisticRegressionModel has not been fitted.")
        # Construct design matrix for prediction
        if features is None:
            # Use combined lagged events as features
            combined = pd.concat([self.train_features_, events.shift(1).fillna(0)], axis=0)
            X_pred = combined.loc[events.index]
        else:
            X_pred = features
        probs = pd.DataFrame(index=events.index, columns=self.contracts_, dtype=float)
        for contract, model in self.models_.items():
            proba = model.predict_proba(X_pred.values)[:, 1]
            probs[contract] = proba
        return probs