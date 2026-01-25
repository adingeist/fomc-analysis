"""
Earnings Call Analysis Framework with Kalshi Integration.

This module provides tools for analyzing earnings call transcripts to predict
Kalshi mention contract outcomes and identify profitable trading opportunities.

Key features:
- Transcript fetching from multiple sources (SEC EDGAR, Alpha Vantage, etc.)
- Speaker segmentation (CEO, CFO, Analysts)
- Sentiment and keyword extraction
- Kalshi contract analysis and outcome prediction
- Walk-forward backtesting for Kalshi mention contracts
"""

from .config import get_config, EarningsConfig
from .parsing import segment_earnings_transcript, parse_transcript
from .features import featurize_earnings_calls
from .fetchers import fetch_earnings_data

# Kalshi integration
from .kalshi import (
    EarningsContractWord,
    EarningsMentionAnalysis,
    EarningsContractAnalyzer,
    analyze_earnings_kalshi_contracts,
)

from .kalshi.backtester import (
    EarningsKalshiBacktester,
    save_earnings_backtest_result,
    create_market_prices_from_tracker,
    run_backtest_with_historical_prices,
)

# Price tracking
from .fetchers.kalshi_price_tracker import (
    KalshiPriceTracker,
    PriceEvolutionAnalyzer,
    record_daily_prices,
)

# Models
from .models import BetaBinomialEarningsModel

__version__ = "0.1.0"

__all__ = [
    "get_config",
    "EarningsConfig",
    "segment_earnings_transcript",
    "parse_transcript",
    "featurize_earnings_calls",
    "fetch_earnings_data",
    "EarningsContractWord",
    "EarningsMentionAnalysis",
    "EarningsContractAnalyzer",
    "analyze_earnings_kalshi_contracts",
    "EarningsKalshiBacktester",
    "save_earnings_backtest_result",
    "create_market_prices_from_tracker",
    "run_backtest_with_historical_prices",
    "BetaBinomialEarningsModel",
    # Price tracking
    "KalshiPriceTracker",
    "PriceEvolutionAnalyzer",
    "record_daily_prices",
]
