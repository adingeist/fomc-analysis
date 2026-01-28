"""Backtest orchestration service for the API layer."""

from __future__ import annotations

import json
import logging
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Optional

from earnings_analysis.kalshi.backtester import (
    BacktestResult,
    EarningsKalshiBacktester,
    save_earnings_backtest_result,
)
from earnings_analysis.models.market_adjusted_model import MarketAdjustedModel

from .model_manager import DEFAULT_MODEL_PARAMS, ModelManager

logger = logging.getLogger(__name__)

BACKTEST_RESULTS_DIR = Path("data/backtest_results")


def run_backtest(
    ticker: str,
    model_manager: ModelManager,
    edge_threshold: float = 0.12,
    initial_capital: float = 10000.0,
) -> Dict[str, Any]:
    """Run a walk-forward backtest for a ticker and return serialisable results."""
    ticker = ticker.upper()

    features, outcomes = model_manager.get_training_data(ticker)

    backtester = EarningsKalshiBacktester(
        features=features,
        outcomes=outcomes,
        model_class=MarketAdjustedModel,
        model_params=DEFAULT_MODEL_PARAMS,
        edge_threshold=edge_threshold,
    )
    result = backtester.run(ticker=ticker, initial_capital=initial_capital)

    # Persist to disk
    out_dir = BACKTEST_RESULTS_DIR / ticker
    save_earnings_backtest_result(result, out_dir)

    return _serialise_result(ticker, result)


def load_backtest(ticker: str) -> Optional[Dict[str, Any]]:
    """Load the most recent persisted backtest result for a ticker."""
    ticker = ticker.upper()
    results_file = BACKTEST_RESULTS_DIR / ticker / "backtest_results.json"

    if not results_file.exists():
        return None

    try:
        data = json.loads(results_file.read_text())
        return {
            "ticker": ticker,
            "metrics": data.get("metrics", {}),
            "trades": data.get("trades", []),
            "metadata": data.get("metadata", {}),
        }
    except Exception:
        logger.warning("Corrupt backtest file for %s", ticker, exc_info=True)
        return None


def _serialise_result(ticker: str, result: BacktestResult) -> Dict[str, Any]:
    return {
        "ticker": ticker,
        "metrics": asdict(result.metrics),
        "trades": [asdict(t) for t in result.trades],
        "metadata": result.metadata,
    }
