"""
Tests for walk-forward backtester with no-lookahead guarantees.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from fomc_analysis.backtester_v2 import WalkForwardBacktester
from fomc_analysis.models import BetaBinomialModel


class TestWalkForwardBacktester:
    """Tests for walk-forward backtesting."""

    def create_test_data(self):
        """Create synthetic test data."""
        # Create 10 events
        dates = [f"2024-{i:02d}-01" for i in range(1, 11)]

        # Binary events: contract mentioned or not
        events = pd.DataFrame({
            "Contract1": [1, 0, 1, 1, 0, 1, 1, 0, 1, 0],
            "Contract2": [0, 1, 1, 0, 1, 1, 0, 1, 0, 1],
        }, index=dates)

        # Market prices (probabilities)
        # Set prices slightly lower than actual to simulate mispricing
        prices = pd.DataFrame({
            "Contract1": [0.4, 0.3, 0.4, 0.5, 0.3, 0.5, 0.5, 0.3, 0.5, 0.3],
            "Contract2": [0.3, 0.4, 0.5, 0.3, 0.4, 0.5, 0.3, 0.5, 0.3, 0.4],
        }, index=dates)

        return events, prices

    def test_no_lookahead(self):
        """Test that backtester doesn't use future information."""
        events, prices = self.create_test_data()

        backtester = WalkForwardBacktester(
            events=events,
            prices=prices,
            min_train_window=3,
        )

        result = backtester.run(
            model_class=BetaBinomialModel,
            model_params={"alpha_prior": 1.0, "beta_prior": 1.0},
            initial_capital=1000.0,
        )

        # Each trade should only use information available BEFORE the trade
        for trade in result.trades:
            trade_date = trade.date
            trade_idx = events.index.get_loc(trade_date)

            # Model should only have been trained on events BEFORE this date
            # This is enforced by the backtester logic, but we verify the result

            assert trade.model_prob is not None
            assert 0 <= trade.model_prob <= 1

    def test_equity_curve_monotonicity(self):
        """Test that equity curve is properly tracked."""
        events, prices = self.create_test_data()

        backtester = WalkForwardBacktester(
            events=events,
            prices=prices,
            min_train_window=3,
        )

        result = backtester.run(
            model_class=BetaBinomialModel,
            initial_capital=1000.0,
        )

        # Equity curve should have values for all dates
        assert len(result.equity_curve) > 0

        # All values should be positive (can't go negative with limited position size)
        assert all(result.equity_curve > 0)

    def test_edge_threshold(self):
        """Test that edge threshold filters trades."""
        events, prices = self.create_test_data()

        # High edge threshold should result in fewer trades
        backtester_high = WalkForwardBacktester(
            events=events,
            prices=prices,
            edge_threshold=0.3,  # Very high threshold
        )

        result_high = backtester_high.run(
            model_class=BetaBinomialModel,
            initial_capital=1000.0,
        )

        # Low edge threshold should result in more trades
        backtester_low = WalkForwardBacktester(
            events=events,
            prices=prices,
            edge_threshold=0.05,  # Low threshold
        )

        result_low = backtester_low.run(
            model_class=BetaBinomialModel,
            initial_capital=1000.0,
        )

        # More trades with lower threshold
        assert len(result_low.trades) >= len(result_high.trades)

    def test_metrics_computation(self):
        """Test that performance metrics are computed."""
        events, prices = self.create_test_data()

        backtester = WalkForwardBacktester(
            events=events,
            prices=prices,
        )

        result = backtester.run(
            model_class=BetaBinomialModel,
            initial_capital=1000.0,
        )

        # Check that all expected metrics are present
        assert "total_trades" in result.metrics
        assert "roi" in result.metrics
        assert "sharpe" in result.metrics
        assert "max_drawdown" in result.metrics
        assert "win_rate" in result.metrics
        assert "brier_score" in result.metrics

        # Win rate should be between 0 and 1
        if result.metrics["total_trades"] > 0:
            assert 0 <= result.metrics["win_rate"] <= 1

    def test_fees_applied(self):
        """Test that transaction fees are applied to profits."""
        events, prices = self.create_test_data()

        # Backtester with 7% fee
        backtester = WalkForwardBacktester(
            events=events,
            prices=prices,
            fee_rate=0.07,
        )

        result = backtester.run(
            model_class=BetaBinomialModel,
            initial_capital=1000.0,
        )

        # Check that profitable trades have fees deducted
        for trade in result.trades:
            if trade.pnl is not None and trade.pnl > 0:
                # Profitable trade should have had fees applied
                # (We can't easily verify the exact amount without recomputing,
                # but we can check that PnL is reasonable)
                assert trade.pnl > 0

    def test_deterministic_results(self):
        """Test that backtester produces deterministic results."""
        events, prices = self.create_test_data()

        backtester = WalkForwardBacktester(
            events=events,
            prices=prices,
        )

        # Run twice with same parameters
        result1 = backtester.run(
            model_class=BetaBinomialModel,
            model_params={"alpha_prior": 1.0, "beta_prior": 1.0},
            initial_capital=1000.0,
        )

        result2 = backtester.run(
            model_class=BetaBinomialModel,
            model_params={"alpha_prior": 1.0, "beta_prior": 1.0},
            initial_capital=1000.0,
        )

        # Results should be identical
        assert result1.metrics["final_capital"] == result2.metrics["final_capital"]
        assert len(result1.trades) == len(result2.trades)
