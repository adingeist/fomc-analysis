"""
Market microstructure tools for Kalshi prediction market trading.

Based on findings from "The Microstructure of Wealth Transfer in Prediction
Markets" (Becker, 2025), which analyzed 72.1M trades ($18.26B volume) on Kalshi.

Key modules:
- calibration: Price-to-win-rate calibration and edge correction
- execution: Order placement optimization (maker vs taker, spread-aware)
- statistical_tests: Significance testing for trading edge claims
- trade_storage: Parquet-based storage for trade and market data
- trade_fetcher: Kalshi API trade data fetching with checkpoint/resume
- trade_analyzer: DuckDB analytical queries on trade data
- regime: Market efficiency monitoring and adaptive thresholds
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

from .trade_storage import (
    ParquetStorage,
    TradeRecord,
    MarketRecord,
)

from .trade_fetcher import (
    EarningsTradesFetcher,
    EARNINGS_TICKERS,
)

from .regime import (
    EfficiencyMonitor,
    EfficiencyMetrics,
    AdaptiveThresholds,
    NewContractDetector,
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
    # Trade data pipeline
    "ParquetStorage",
    "TradeRecord",
    "MarketRecord",
    "EarningsTradesFetcher",
    "EARNINGS_TICKERS",
    # Regime monitoring
    "EfficiencyMonitor",
    "EfficiencyMetrics",
    "AdaptiveThresholds",
    "NewContractDetector",
]
