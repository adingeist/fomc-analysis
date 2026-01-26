"""
Market microstructure tools for Kalshi prediction market trading.

Based on findings from "The Microstructure of Wealth Transfer in Prediction
Markets" (Becker, 2025), which analyzed 72.1M trades ($18.26B volume) on Kalshi.

Key modules:
- calibration: Price-to-win-rate calibration and edge correction
- execution: Order placement optimization (maker vs taker, spread-aware)
- statistical_tests: Significance testing for trading edge claims
"""

from .calibration import (
    KalshiCalibrationCurve,
    calibrated_probability,
    calibrated_edge,
    directional_bias_score,
)

from .execution import (
    ExecutionSimulator,
    SpreadFilter,
    ExecutionMode,
)

from .statistical_tests import (
    test_edge_significance,
    test_calibration,
    test_multiple_edges,
    compute_brier_decomposition,
    SignificanceResult,
    CalibrationResult,
)

__all__ = [
    # Calibration
    "KalshiCalibrationCurve",
    "calibrated_probability",
    "calibrated_edge",
    "directional_bias_score",
    # Execution
    "ExecutionSimulator",
    "SpreadFilter",
    "ExecutionMode",
    # Statistical tests
    "test_edge_significance",
    "test_calibration",
    "test_multiple_edges",
    "compute_brier_decomposition",
    "SignificanceResult",
    "CalibrationResult",
]
