"""Find edge opportunities by comparing model predictions to live market prices."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from fomc_analysis.kalshi_client_factory import KalshiClientProtocol

from .model_manager import ModelManager

logger = logging.getLogger(__name__)

# Series ticker prefix
_SERIES_PREFIX = "KXEARNINGSMENTION"


def _extract_word_from_market(market: Dict[str, Any]) -> Optional[str]:
    """Extract the tracked word from a Kalshi market dict."""
    custom_strike = market.get("custom_strike") or {}
    word = custom_strike.get("Word") or market.get("yes_sub_title", "")
    return word.strip() if word else None


def find_edges_for_ticker(
    ticker: str,
    client: KalshiClientProtocol,
    model_manager: ModelManager,
    min_edge: float = 0.05,
    signal_filter: Optional[str] = None,
) -> Dict[str, Any]:
    """Find edge opportunities for a single ticker.

    Returns a dict with ``opportunities`` (list) and ``total_scanned`` (int).
    """
    ticker = ticker.upper()
    series_ticker = f"{_SERIES_PREFIX}{ticker}"

    markets = client.get_markets(series_ticker=series_ticker, status="open")
    if not markets:
        markets = client.get_markets(series_ticker=series_ticker)

    opportunities: List[Dict[str, Any]] = []
    scanned = 0

    for market in markets:
        status = (market.get("status") or "").lower()
        if status not in ("active", "open", ""):
            continue

        word = _extract_word_from_market(market)
        if not word:
            continue

        scanned += 1

        # Market price in 0-1 scale
        last_price = market.get("last_price")
        if last_price is None:
            continue
        market_price = last_price / 100.0

        yes_bid_raw = market.get("yes_bid")
        yes_ask_raw = market.get("yes_ask")
        yes_bid = yes_bid_raw / 100.0 if yes_bid_raw is not None else None
        yes_ask = yes_ask_raw / 100.0 if yes_ask_raw is not None else None

        # Get model prediction
        word_key = word.lower()
        try:
            pred = model_manager.predict(ticker, word_key, market_price=market_price)
        except KeyError:
            continue

        edge = pred.get("edge")
        adjusted_edge = pred.get("adjusted_edge")
        signal = pred.get("trade_signal", "HOLD")

        if edge is None or adjusted_edge is None:
            continue

        # Filter by minimum edge
        if abs(adjusted_edge) < min_edge:
            continue

        # Filter by signal direction
        if signal_filter and signal != signal_filter:
            continue

        if signal == "HOLD":
            continue

        opportunities.append(
            {
                "ticker": ticker,
                "word": word,
                "predicted_probability": pred.get("probability", 0.0),
                "market_price": market_price,
                "edge": edge,
                "adjusted_edge": adjusted_edge,
                "signal": signal,
                "kelly_fraction": pred.get("kelly_fraction", 0.0),
                "confidence": pred.get("confidence"),
                "yes_bid": yes_bid,
                "yes_ask": yes_ask,
                "market_ticker": market.get("ticker", ""),
            }
        )

    # Sort by absolute adjusted edge descending
    opportunities.sort(key=lambda o: abs(o["adjusted_edge"]), reverse=True)

    return {"opportunities": opportunities, "total_scanned": scanned}


def find_edges_all_tickers(
    client: KalshiClientProtocol,
    model_manager: ModelManager,
    min_edge: float = 0.05,
    signal_filter: Optional[str] = None,
) -> Dict[str, Any]:
    """Find edges across all tickers with trained models."""
    all_opportunities: List[Dict[str, Any]] = []
    total_scanned = 0

    for ticker in model_manager.models:
        try:
            result = find_edges_for_ticker(
                ticker, client, model_manager, min_edge, signal_filter
            )
            all_opportunities.extend(result["opportunities"])
            total_scanned += result["total_scanned"]
        except Exception:
            logger.warning("Error scanning edges for %s", ticker, exc_info=True)

    all_opportunities.sort(key=lambda o: abs(o["adjusted_edge"]), reverse=True)

    return {"opportunities": all_opportunities, "total_scanned": total_scanned}
