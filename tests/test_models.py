"""
Tests for probability models and uncertainty estimation.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile

from fomc_analysis.models import EWMAModel, BetaBinomialModel


class TestEWMAModel:
    """Tests for EWMA model."""

    def test_fit_and_predict(self):
        """Test basic fit and predict."""
        # Create simple binary event data
        events = pd.DataFrame({
            "Contract1": [1, 0, 1, 1, 0],
            "Contract2": [0, 0, 1, 1, 1],
        })

        model = EWMAModel(alpha=0.5)
        model.fit(events)

        predictions = model.predict()

        assert len(predictions) == 2
        assert "probability" in predictions.columns
        assert "lower_bound" in predictions.columns
        assert "upper_bound" in predictions.columns
        assert "uncertainty" in predictions.columns

        # Probabilities should be between 0 and 1
        assert all(0 <= p <= 1 for p in predictions["probability"])

    def test_save_and_load(self):
        """Test model persistence."""
        events = pd.DataFrame({
            "Contract1": [1, 0, 1, 1, 0],
        })

        model = EWMAModel(alpha=0.6)
        model.fit(events)

        np.random.seed(0)
        pred_before = model.predict()

        # Save and load
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "model.json"
            model.save(path)

            loaded_model = EWMAModel.load(path)
            np.random.seed(0)
            pred_after = loaded_model.predict()

        # Predictions should be identical
        pd.testing.assert_frame_equal(pred_before, pred_after)


class TestBetaBinomialModel:
    """Tests for Beta-Binomial model."""

    def test_fit_and_predict(self):
        """Test basic fit and predict."""
        events = pd.DataFrame({
            "Contract1": [1, 0, 1, 1, 0],
            "Contract2": [0, 0, 1, 1, 1],
        })

        model = BetaBinomialModel(alpha_prior=1.0, beta_prior=1.0)
        model.fit(events)

        predictions = model.predict()

        assert len(predictions) == 2
        assert "probability" in predictions.columns
        assert "lower_bound" in predictions.columns
        assert "upper_bound" in predictions.columns
        assert "uncertainty" in predictions.columns

        # Probabilities should be between 0 and 1
        assert all(0 <= p <= 1 for p in predictions["probability"])

        # Lower bound should be less than upper bound
        for _, row in predictions.iterrows():
            assert row["lower_bound"] <= row["probability"] <= row["upper_bound"]

    def test_uniform_prior(self):
        """Test that uniform prior (alpha=1, beta=1) gives reasonable results."""
        # All events are 1s
        events = pd.DataFrame({
            "AlwaysYes": [1, 1, 1, 1, 1],
        })

        model = BetaBinomialModel(alpha_prior=1.0, beta_prior=1.0)
        model.fit(events)

        predictions = model.predict()

        # With 5 successes and 0 failures, probability should be high
        assert predictions.iloc[0]["probability"] > 0.7

    def test_exponential_decay(self):
        """Test exponential decay weighting."""
        # Recent events are 1, older events are 0
        events = pd.DataFrame({
            "Contract": [0, 0, 0, 1, 1],
        })

        # Without decay
        model_no_decay = BetaBinomialModel(half_life=None)
        model_no_decay.fit(events)
        pred_no_decay = model_no_decay.predict()

        # With decay (half_life=2 means recent events weighted more)
        model_decay = BetaBinomialModel(half_life=2)
        model_decay.fit(events)
        pred_decay = model_decay.predict()

        # With decay, probability should be higher (recent events are 1s)
        assert pred_decay.iloc[0]["probability"] > pred_no_decay.iloc[0]["probability"]

    def test_handles_missing_contract_history(self):
        """Model should ignore NaNs rather than propagating them."""
        events = pd.DataFrame({
            "Contract1": [1, np.nan, 0, np.nan],
            "Contract2": [np.nan, np.nan, np.nan, np.nan],
        })

        model = BetaBinomialModel(alpha_prior=1.0, beta_prior=1.0, half_life=2)
        model.fit(events)

        predictions = model.predict()

        contract1 = predictions[predictions["contract"] == "Contract1"].iloc[0]
        assert np.isfinite(contract1["probability"])
        assert 0 <= contract1["probability"] <= 1

        # With no historical data, the model should revert to the prior (0.5 for alpha=beta=1)
        contract2 = predictions[predictions["contract"] == "Contract2"].iloc[0]
        assert contract2["probability"] == pytest.approx(0.5)

    def test_prior_strength_uses_contract_base_rates(self):
        """Contract-specific priors should pull frequent mentions higher."""
        events = pd.DataFrame({
            "Often": [1, 1, 1, 1, 1],
            "Rare": [1, 0, 0, 0, 0],
        })

        model = BetaBinomialModel(prior_strength=10.0, min_history=3)
        model.fit(events)

        predictions = model.predict()
        often_prob = predictions[predictions["contract"] == "Often"].iloc[0]["probability"]
        rare_prob = predictions[predictions["contract"] == "Rare"].iloc[0]["probability"]

        assert often_prob > rare_prob
        assert rare_prob < 0.5

    def test_min_history_shrinks_sparse_contracts(self):
        events = pd.DataFrame({
            "Sparse": [1, np.nan, np.nan, np.nan],
        })

        model = BetaBinomialModel(min_history=4)
        model.fit(events)

        prob = model.predict().iloc[0]["probability"]
        assert prob == pytest.approx(0.5, abs=0.1)

    def test_save_and_load(self):
        """Test model persistence."""
        events = pd.DataFrame({
            "Contract1": [1, 0, 1, 1, 0],
        })

        model = BetaBinomialModel(alpha_prior=2.0, beta_prior=1.0, half_life=3)
        model.fit(events)

        pred_before = model.predict()

        # Save and load
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "model.json"
            model.save(path)

            loaded_model = BetaBinomialModel.load(path)
            pred_after = loaded_model.predict()

        # Predictions should be identical
        pd.testing.assert_frame_equal(pred_before, pred_after)
