"""Kalshi integration for earnings call analysis."""

from .contract_analyzer import (
    EarningsContractWord,
    EarningsMentionAnalysis,
    EarningsContractAnalyzer,
    analyze_earnings_kalshi_contracts,
)

from .backtester import (
    EarningsKalshiBacktester,
    EarningsPrediction,
    Trade,
    BacktestMetrics,
    BacktestResult,
    save_earnings_backtest_result,
    compute_backtest_significance,
    create_market_prices_from_tracker,
    run_backtest_with_historical_prices,
)

__all__ = [
    "EarningsContractWord",
    "EarningsMentionAnalysis",
    "EarningsContractAnalyzer",
    "analyze_earnings_kalshi_contracts",
    # Backtester
    "EarningsKalshiBacktester",
    "EarningsPrediction",
    "Trade",
    "BacktestMetrics",
    "BacktestResult",
    "save_earnings_backtest_result",
    "compute_backtest_significance",
    "create_market_prices_from_tracker",
    "run_backtest_with_historical_prices",
]
