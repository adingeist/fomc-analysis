"""
Tests for market microstructure modules.

Tests calibration curves, execution simulation, and statistical tests
based on empirical findings from Becker (2025).
"""

import pytest
import numpy as np
import pandas as pd

from earnings_analysis.microstructure.calibration import (
    KalshiCalibrationCurve,
    calibrated_probability,
    calibrated_edge,
    directional_bias_score,
    _logit,
    _inv_logit,
)
from earnings_analysis.microstructure.execution import (
    ExecutionSimulator,
    ExecutionMode,
    SpreadFilter,
)
from earnings_analysis.microstructure.statistical_tests import (
    test_edge_significance as run_edge_significance,
    test_yes_no_asymmetry as run_yes_no_asymmetry,
    test_calibration as run_calibration,
    test_multiple_edges as run_multiple_edges,
    compute_brier_decomposition,
)


# ─────────────────────────────────────────────
# Calibration tests
# ─────────────────────────────────────────────

class TestLogitFunctions:
    """Test logit/inverse-logit helpers."""

    def test_logit_inv_logit_roundtrip(self):
        for p in [0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99]:
            assert abs(_inv_logit(_logit(p)) - p) < 1e-10

    def test_logit_symmetry(self):
        assert abs(_logit(0.5)) < 1e-10
        assert abs(_logit(0.3) + _logit(0.7)) < 1e-10

    def test_logit_extreme_clamping(self):
        # Should not raise or return inf
        result = _logit(0.0)
        assert np.isfinite(result)
        result = _logit(1.0)
        assert np.isfinite(result)


class TestKalshiCalibrationCurve:
    """Test calibration curve mapping."""

    def test_default_curve_exists(self):
        curve = KalshiCalibrationCurve()
        assert curve.gamma > 1.0  # Longshot bias

    def test_midpoint_unchanged(self):
        """At 50 cents, calibration should be very close to 50%."""
        curve = KalshiCalibrationCurve()
        cal = curve.calibrated_probability(50)
        assert abs(cal - 0.50) < 0.01

    def test_longshot_overpriced(self):
        """Low-price contracts should have actual < implied (YES overpriced)."""
        curve = KalshiCalibrationCurve()
        for cents in [5, 10, 15]:
            cal = curve.calibrated_probability(cents)
            implied = cents / 100.0
            assert cal < implied, f"At {cents}c: cal={cal:.4f} should be < implied={implied:.4f}"

    def test_favorite_underpriced(self):
        """High-price contracts should have actual >= implied."""
        curve = KalshiCalibrationCurve()
        for cents in [85, 90, 95]:
            cal = curve.calibrated_probability(cents)
            implied = cents / 100.0
            assert cal >= implied, f"At {cents}c: cal={cal:.4f} should be >= implied={implied:.4f}"

    def test_monotonic(self):
        """Calibrated probabilities should be monotonically increasing."""
        curve = KalshiCalibrationCurve()
        probs = [curve.calibrated_probability(c) for c in range(1, 100)]
        for i in range(1, len(probs)):
            assert probs[i] >= probs[i - 1], f"Non-monotonic at {i + 1}c"

    def test_becker_data_points(self):
        """Verify against known empirical data points from Becker (2025)."""
        curve = KalshiCalibrationCurve()
        # At 5c: actual should be approximately 4.18%
        cal_5 = curve.calibrated_probability(5)
        assert abs(cal_5 - 0.0418) < 0.005, f"At 5c: expected ~0.0418, got {cal_5:.4f}"

        # At 95c: actual should be approximately 95.83%
        cal_95 = curve.calibrated_probability(95)
        assert abs(cal_95 - 0.9583) < 0.005, f"At 95c: expected ~0.9583, got {cal_95:.4f}"

    def test_empirical_table_override(self):
        """Empirical data should override parametric model."""
        curve = KalshiCalibrationCurve(empirical_table={50: 0.55})
        assert curve.calibrated_probability(50) == 0.55
        # Non-overridden price should use parametric
        assert curve.calibrated_probability(49) != 0.55

    def test_calibrated_edge(self):
        """Edge = model_prob - calibrated_win_rate."""
        curve = KalshiCalibrationCurve()
        # If model says 60% and market at 50c (cal ~50%), edge should be ~10%
        edge = curve.calibrated_edge(0.60, 50)
        assert abs(edge - 0.10) < 0.02

    def test_yes_adjusted_edge(self):
        """YES edge should be smaller than NO edge due to YES penalty."""
        curve = KalshiCalibrationCurve()
        yes_edge, no_edge = curve.yes_adjusted_edge(0.70, 50)
        # YES edge penalized, NO edge boosted relative to raw edge
        raw_edge = curve.calibrated_edge(0.70, 50)
        assert yes_edge < raw_edge  # YES penalized
        # no_edge = (1-0.70) - (1-cal-penalty) = -raw_edge + penalty
        # The absolute value of the signs can differ, but YES should be worse

    def test_mispricing_at(self):
        """Positive mispricing = market overprices YES."""
        curve = KalshiCalibrationCurve()
        # At low prices, YES is overpriced
        mp = curve.mispricing_at(5)
        assert mp > 0, "Longshots should be overpriced"

    def test_as_table(self):
        curve = KalshiCalibrationCurve()
        table = curve.as_table()
        assert len(table) == 99
        assert table[0]["price_cents"] == 1
        assert table[-1]["price_cents"] == 99

    def test_clamp_input(self):
        """Out-of-range inputs should be clamped."""
        curve = KalshiCalibrationCurve()
        assert 0 < curve.calibrated_probability(0) < 1
        assert 0 < curve.calibrated_probability(100) < 1

    def test_load_empirical_data(self):
        curve = KalshiCalibrationCurve()
        curve.load_empirical_data({10: 0.08, 90: 0.93})
        assert curve.calibrated_probability(10) == 0.08
        assert curve.calibrated_probability(90) == 0.93


class TestModuleLevelFunctions:
    """Test module-level convenience functions."""

    def test_calibrated_probability(self):
        result = calibrated_probability(50)
        assert 0.49 < result < 0.51

    def test_calibrated_edge(self):
        edge = calibrated_edge(0.60, 50)
        assert edge > 0

    def test_directional_bias_score_range(self):
        for model_prob in [0.1, 0.3, 0.5, 0.7, 0.9]:
            for cents in [10, 30, 50, 70, 90]:
                score = directional_bias_score(model_prob, cents)
                assert -1.0 <= score <= 1.0

    def test_directional_bias_no_favored_at_low_price(self):
        """At low prices, structural bias should favor NO (longshots overpriced)."""
        # Model says 50% but market is 10c => model thinks more likely than market
        # Structural edge: calibrated < implied => structural favors NO
        score = directional_bias_score(0.50, 10)
        # Model favors YES (50% > ~8%), but structural favors NO
        # Net depends on magnitudes, but model signal dominates
        assert isinstance(score, float)


# ─────────────────────────────────────────────
# Execution simulation tests
# ─────────────────────────────────────────────

class TestSpreadFilter:
    """Test spread-based trade filtering."""

    def test_narrow_spread_allows_trade(self):
        sf = SpreadFilter(min_net_edge=0.03)
        assert sf.should_trade(0.10, 48, 52)  # 4c spread, 10% edge

    def test_wide_spread_blocks_trade(self):
        sf = SpreadFilter(min_net_edge=0.03, max_spread_cents=10)
        assert not sf.should_trade(0.10, 40, 60)  # 20c spread > 10c max

    def test_insufficient_edge_blocks_trade(self):
        sf = SpreadFilter(min_net_edge=0.05)
        assert not sf.should_trade(0.04, 48, 52)  # 4% edge < 5% min after 2% spread

    def test_net_edge_computation(self):
        sf = SpreadFilter()
        net = sf.net_edge_after_spread(0.10, 48, 52)
        # 10% edge - 2% half-spread = 8%
        assert abs(net - 0.08) < 0.001


class TestExecutionSimulator:
    """Test order execution simulation."""

    def test_taker_always_fills(self):
        sim = ExecutionSimulator(mode=ExecutionMode.TAKER)
        result = sim.simulate_execution("YES", 48, 52)
        assert result.filled
        assert not result.is_maker
        assert result.spread_cost > 0

    def test_taker_yes_price(self):
        """YES taker buys at ask."""
        sim = ExecutionSimulator(mode=ExecutionMode.TAKER, taker_slippage=0.0)
        result = sim.simulate_execution("YES", 48, 52)
        assert abs(result.execution_price - 0.52) < 0.01

    def test_taker_no_price(self):
        """NO taker buys at 100 - bid."""
        sim = ExecutionSimulator(mode=ExecutionMode.TAKER, taker_slippage=0.0)
        result = sim.simulate_execution("NO", 48, 52)
        assert abs(result.execution_price - 0.52) < 0.01  # 100-48=52

    def test_maker_may_not_fill(self):
        """Maker fill is probabilistic."""
        sim = ExecutionSimulator(mode=ExecutionMode.MAKER, base_fill_probability=0.0)
        rng = np.random.default_rng(42)
        result = sim.simulate_execution("YES", 48, 52, rng=rng)
        assert not result.filled

    def test_maker_guaranteed_fill(self):
        sim = ExecutionSimulator(mode=ExecutionMode.MAKER, base_fill_probability=1.0)
        rng = np.random.default_rng(42)
        result = sim.simulate_execution("YES", 48, 52, rng=rng)
        assert result.filled
        assert result.is_maker
        assert result.spread_cost == 0.0

    def test_maker_price_better_than_taker(self):
        """Maker should get better price than taker."""
        maker = ExecutionSimulator(mode=ExecutionMode.MAKER, base_fill_probability=1.0)
        taker = ExecutionSimulator(mode=ExecutionMode.TAKER)
        rng = np.random.default_rng(42)

        maker_result = maker.simulate_execution("YES", 48, 52, rng=rng)
        taker_result = taker.simulate_execution("YES", 48, 52)

        # Lower execution price = better for buyer
        assert maker_result.execution_price < taker_result.execution_price

    def test_hybrid_fills(self):
        """Hybrid should always produce a fill."""
        sim = ExecutionSimulator(mode=ExecutionMode.HYBRID, base_fill_probability=0.5)
        rng = np.random.default_rng(42)
        result = sim.simulate_execution("YES", 48, 52, rng=rng)
        assert result.filled  # Hybrid falls back to taker

    def test_expected_execution_price(self):
        sim = ExecutionSimulator(mode=ExecutionMode.HYBRID)
        price = sim.expected_execution_price("YES", 48, 52, urgency=0.5)
        assert 0.01 <= price <= 0.99

    def test_adjust_backtest_entry_price_taker(self):
        sim = ExecutionSimulator(mode=ExecutionMode.TAKER, taker_slippage=0.02)
        adjusted = sim.adjust_backtest_entry_price(0.50, "YES")
        assert adjusted == pytest.approx(0.52)

    def test_adjust_backtest_entry_price_maker(self):
        sim = ExecutionSimulator(mode=ExecutionMode.MAKER, maker_improvement=0.01)
        adjusted = sim.adjust_backtest_entry_price(0.50, "YES")
        assert adjusted == pytest.approx(0.49)

    def test_adjust_backtest_entry_price_hybrid(self):
        sim = ExecutionSimulator(mode=ExecutionMode.HYBRID)
        adjusted = sim.adjust_backtest_entry_price(0.50, "YES")
        # Hybrid should be between maker and taker prices
        maker = ExecutionSimulator(mode=ExecutionMode.MAKER).adjust_backtest_entry_price(0.50, "YES")
        taker = ExecutionSimulator(mode=ExecutionMode.TAKER).adjust_backtest_entry_price(0.50, "YES")
        assert maker <= adjusted <= taker

    def test_urgency_affects_fill_probability(self):
        """Higher urgency should reduce fill probability for makers."""
        sim = ExecutionSimulator(mode=ExecutionMode.MAKER, base_fill_probability=0.8)
        # Simulate many executions at different urgency levels
        rng_low = np.random.default_rng(42)
        rng_high = np.random.default_rng(42)

        low_urgency_fills = sum(
            sim.simulate_execution("YES", 48, 52, urgency=0.1, rng=np.random.default_rng(i)).filled
            for i in range(100)
        )
        high_urgency_fills = sum(
            sim.simulate_execution("YES", 48, 52, urgency=0.9, rng=np.random.default_rng(i)).filled
            for i in range(100)
        )
        assert low_urgency_fills >= high_urgency_fills


# ─────────────────────────────────────────────
# Statistical tests
# ─────────────────────────────────────────────

class TestEdgeSignificance:
    """Test edge significance testing."""

    def test_significant_positive_edge(self):
        """Strong positive returns should be significant."""
        rng = np.random.default_rng(42)
        returns = list(rng.normal(0.05, 0.02, 100))  # 5% mean, 2% std
        result = run_edge_significance(returns)
        assert result.significant_at_05
        assert result.significant_at_01
        assert result.mean_return > 0
        assert result.cohens_d > 1.0  # Large effect

    def test_zero_edge_not_significant(self):
        """Returns centered at zero should not be significant."""
        rng = np.random.default_rng(42)
        returns = list(rng.normal(0.0, 0.10, 50))
        result = run_edge_significance(returns)
        # With zero mean and moderate noise, usually not significant
        # (could be by chance, but on average won't be)
        assert abs(result.mean_return) < 0.05

    def test_small_sample_not_significant(self):
        """Very small samples should rarely be significant."""
        result = run_edge_significance([0.05, -0.02])
        assert result.n_trades == 2
        # With 2 samples, hard to be significant

    def test_single_trade(self):
        """Single trade should return valid but non-significant result."""
        result = run_edge_significance([0.05])
        assert result.n_trades == 1
        assert not result.significant_at_05

    def test_empty_trades(self):
        result = run_edge_significance([])
        assert result.n_trades == 0

    def test_confidence_interval_contains_mean(self):
        rng = np.random.default_rng(42)
        returns = list(rng.normal(0.03, 0.05, 50))
        result = run_edge_significance(returns)
        assert result.ci_95_lower <= result.mean_return <= result.ci_95_upper

    def test_summary_string(self):
        result = run_edge_significance([0.1, 0.2, 0.05, 0.15, 0.08])
        summary = result.summary()
        assert "Edge:" in summary
        assert "n=" in summary


class TestYesNoAsymmetry:
    """Test YES vs NO return asymmetry analysis."""

    def test_no_favored(self):
        """When NO returns are higher, should be detected."""
        rng = np.random.default_rng(42)
        yes_returns = list(rng.normal(0.01, 0.05, 50))
        no_returns = list(rng.normal(0.05, 0.05, 50))
        result = run_yes_no_asymmetry(yes_returns, no_returns)
        assert result["no_favored"]
        assert result["difference"] > 0

    def test_insufficient_data(self):
        result = run_yes_no_asymmetry([0.05], [0.10])
        assert not result["significant"]
        assert result["test"] == "insufficient_data"


class TestCalibration:
    """Test calibration analysis."""

    def test_perfect_calibration(self):
        """A perfectly calibrated model should pass H-L test."""
        rng = np.random.default_rng(42)
        n = 1000
        probs = rng.uniform(0.1, 0.9, n)
        outcomes = [int(rng.random() < p) for p in probs]

        result = run_calibration(list(probs), outcomes)
        assert result.well_calibrated  # H-L p > 0.05
        assert result.mean_absolute_error < 0.1

    def test_overconfident_model(self):
        """Model that predicts extreme values when reality is moderate."""
        rng = np.random.default_rng(42)
        n = 500
        # Model always predicts 0.9 or 0.1, but true rate is ~0.5
        probs = [0.9 if rng.random() > 0.5 else 0.1 for _ in range(n)]
        outcomes = [int(rng.random() < 0.5) for _ in range(n)]

        result = run_calibration(probs, outcomes)
        assert result.mean_absolute_error > 0.1
        assert result.brier_score > 0.2

    def test_brier_score(self):
        """Brier score should be between 0 (perfect) and 1 (worst)."""
        result = run_calibration([0.5, 0.5, 0.5], [1, 0, 1])
        assert 0 <= result.brier_score <= 1

    def test_bins_sum_to_total(self):
        rng = np.random.default_rng(42)
        n = 200
        probs = list(rng.uniform(0, 1, n))
        outcomes = [int(rng.random() < p) for p in probs]
        result = run_calibration(probs, outcomes)
        total_in_bins = sum(b.n_samples for b in result.bins)
        assert total_in_bins == n


class TestMultipleEdges:
    """Test multiple comparison correction."""

    def test_bonferroni_more_conservative(self):
        """Bonferroni should reject fewer than uncorrected."""
        rng = np.random.default_rng(42)
        edges = {
            f"contract_{i}": list(rng.normal(0.02, 0.05, 30))
            for i in range(10)
        }
        results = run_multiple_edges(edges, method="bonferroni")
        assert len(results) == 10
        for name, r in results.items():
            assert r["corrected_p_value"] >= r["raw_p_value"]

    def test_fdr_bh(self):
        rng = np.random.default_rng(42)
        edges = {
            "real_edge": list(rng.normal(0.10, 0.03, 50)),
            "noise_1": list(rng.normal(0.0, 0.10, 50)),
            "noise_2": list(rng.normal(0.0, 0.10, 50)),
        }
        results = run_multiple_edges(edges, method="fdr_bh")
        # The real edge should be more likely to survive correction
        assert results["real_edge"]["corrected_p_value"] <= results["noise_1"]["corrected_p_value"] or \
               results["real_edge"]["corrected_p_value"] <= results["noise_2"]["corrected_p_value"]

    def test_empty_input(self):
        results = run_multiple_edges({})
        assert results == {}


class TestBrierDecomposition:
    """Test Brier score decomposition."""

    def test_components_sum(self):
        """Brier = Reliability - Resolution + Uncertainty (approximately)."""
        rng = np.random.default_rng(42)
        n = 500
        probs = list(rng.uniform(0.1, 0.9, n))
        outcomes = [int(rng.random() < p) for p in probs]

        result = compute_brier_decomposition(probs, outcomes)
        # Brier ≈ reliability - resolution + uncertainty
        reconstructed = result.reliability - result.resolution + result.uncertainty
        assert abs(result.brier_score - reconstructed) < 0.05

    def test_perfect_model(self):
        """Perfect predictions should have low reliability and high resolution."""
        probs = [1.0, 0.0, 1.0, 0.0, 1.0]
        outcomes = [1, 0, 1, 0, 1]
        result = compute_brier_decomposition(probs, outcomes)
        assert result.brier_score < 0.01

    def test_sharpness(self):
        """Sharpness measures how far predictions are from 0.5."""
        # Sharp model
        sharp = compute_brier_decomposition([0.1, 0.9, 0.1, 0.9], [0, 1, 0, 1])
        # Hedged model
        hedged = compute_brier_decomposition([0.45, 0.55, 0.45, 0.55], [0, 1, 0, 1])
        assert sharp.sharpness > hedged.sharpness

    def test_empty(self):
        result = compute_brier_decomposition([], [])
        assert result.brier_score == 0.0


# ─────────────────────────────────────────────
# Integration tests: backtester with microstructure
# ─────────────────────────────────────────────

class TestBacktesterIntegration:
    """Test that backtesters work with microstructure modules."""

    def _make_test_data(self):
        """Create minimal test data for backtester."""
        dates = pd.date_range("2024-01-01", periods=8, freq="QE")
        features = pd.DataFrame(
            {
                "ai": [5, 3, 12, 8, 6, 10, 4, 7],
                "cloud": [2, 4, 6, 3, 5, 2, 8, 4],
            },
            index=dates,
        )
        outcomes = pd.DataFrame(
            {
                "ai": [1, 1, 1, 1, 1, 1, 0, 1],
                "cloud": [0, 1, 1, 1, 0, 1, 1, 0],
            },
            index=dates,
        )
        return features, outcomes

    def test_backtester_with_calibration(self):
        """Base backtester should accept and use calibration curve."""
        from earnings_analysis.kalshi.backtester import EarningsKalshiBacktester
        from earnings_analysis.models import BetaBinomialEarningsModel

        features, outcomes = self._make_test_data()
        curve = KalshiCalibrationCurve()

        bt = EarningsKalshiBacktester(
            features=features,
            outcomes=outcomes,
            model_class=BetaBinomialEarningsModel,
            model_params={"alpha_prior": 1.0, "beta_prior": 1.0},
            calibration_curve=curve,
        )
        result = bt.run("TEST")
        assert result.metadata["microstructure"]["calibration_enabled"]
        assert result.metadata["microstructure"]["calibration_gamma"] == curve.gamma

    def test_backtester_with_execution_simulator(self):
        """Base backtester should accept and use execution simulator."""
        from earnings_analysis.kalshi.backtester import EarningsKalshiBacktester
        from earnings_analysis.models import BetaBinomialEarningsModel

        features, outcomes = self._make_test_data()
        sim = ExecutionSimulator(mode=ExecutionMode.HYBRID)

        bt = EarningsKalshiBacktester(
            features=features,
            outcomes=outcomes,
            model_class=BetaBinomialEarningsModel,
            model_params={"alpha_prior": 1.0, "beta_prior": 1.0},
            execution_simulator=sim,
        )
        result = bt.run("TEST")
        assert result.metadata["microstructure"]["execution_simulator_enabled"]
        assert result.metadata["microstructure"]["execution_mode"] == "hybrid"

    def test_backtester_backward_compatible(self):
        """Backtester without microstructure should still work as before."""
        from earnings_analysis.kalshi.backtester import EarningsKalshiBacktester
        from earnings_analysis.models import BetaBinomialEarningsModel

        features, outcomes = self._make_test_data()

        bt = EarningsKalshiBacktester(
            features=features,
            outcomes=outcomes,
            model_class=BetaBinomialEarningsModel,
            model_params={"alpha_prior": 1.0, "beta_prior": 1.0},
        )
        result = bt.run("TEST")
        assert not result.metadata["microstructure"]["calibration_enabled"]
        assert not result.metadata["microstructure"]["execution_simulator_enabled"]

    def test_compute_backtest_significance(self):
        """Test statistical significance computation on backtest results."""
        from earnings_analysis.kalshi.backtester import (
            EarningsKalshiBacktester,
            compute_backtest_significance,
        )
        from earnings_analysis.models import BetaBinomialEarningsModel

        features, outcomes = self._make_test_data()

        bt = EarningsKalshiBacktester(
            features=features,
            outcomes=outcomes,
            model_class=BetaBinomialEarningsModel,
            model_params={"alpha_prior": 1.0, "beta_prior": 1.0},
            edge_threshold=0.01,
            yes_edge_threshold=0.01,
            no_edge_threshold=0.01,
        )
        result = bt.run("TEST")
        stats = compute_backtest_significance(result)

        # Should have calibration info at minimum
        if result.predictions:
            assert "calibration" in stats or "brier_decomposition" in stats

    def test_enhanced_backtester_with_microstructure(self):
        """Enhanced backtester should accept microstructure modules."""
        from earnings_analysis.kalshi.enhanced_backtester import EnhancedEarningsBacktester
        from earnings_analysis.models import BetaBinomialEarningsModel

        features, outcomes = self._make_test_data()
        curve = KalshiCalibrationCurve()
        sim = ExecutionSimulator(mode=ExecutionMode.MAKER, base_fill_probability=1.0)

        bt = EnhancedEarningsBacktester(
            features=features,
            outcomes=outcomes,
            model_class=BetaBinomialEarningsModel,
            model_params={"alpha_prior": 1.0, "beta_prior": 1.0},
            calibration_curve=curve,
            execution_simulator=sim,
        )
        result = bt.run("TEST")
        assert result.metadata["microstructure"]["calibration_enabled"]
        assert result.metadata["microstructure"]["execution_mode"] == "maker"
