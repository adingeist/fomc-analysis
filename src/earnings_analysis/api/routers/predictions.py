"""Prediction endpoints."""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query

from earnings_analysis.api.dependencies import get_kalshi_client, get_model_manager
from earnings_analysis.api.schemas import PredictionResponse, WordPrediction
from earnings_analysis.api.services.model_manager import ModelManager
from fomc_analysis.kalshi_client_factory import KalshiClientProtocol

router = APIRouter(prefix="/predictions", tags=["predictions"])

_SERIES_PREFIX = "KXEARNINGSMENTION"


def _extract_word_from_market(market: dict) -> Optional[str]:
    custom_strike = market.get("custom_strike") or {}
    word = custom_strike.get("Word") or market.get("yes_sub_title", "")
    return word.strip() if word else None


def _fetch_live_prices(client: KalshiClientProtocol, ticker: str) -> dict[str, float]:
    """Fetch live market prices for all active contracts of a ticker."""
    series_ticker = f"{_SERIES_PREFIX}{ticker.upper()}"
    markets = client.get_markets(series_ticker=series_ticker, status="open")
    if not markets:
        markets = client.get_markets(series_ticker=series_ticker)

    prices: dict[str, float] = {}
    for market in markets:
        word = _extract_word_from_market(market)
        if not word:
            continue
        last_price = market.get("last_price")
        if last_price is not None:
            prices[word.lower()] = last_price / 100.0
    return prices


@router.get("/{ticker}", response_model=PredictionResponse)
async def get_predictions(
    ticker: str,
    client: KalshiClientProtocol = Depends(get_kalshi_client),
    model_manager: ModelManager = Depends(get_model_manager),
    include_market_prices: bool = Query(
        True, description="Fetch live market prices from Kalshi"
    ),
):
    """Get predictions for all tracked words for a ticker."""
    ticker = ticker.upper()

    if ticker not in model_manager.models:
        raise HTTPException(
            status_code=404,
            detail=f"No models for ticker {ticker}. Available: {list(model_manager.models.keys())}",
        )

    market_prices: dict[str, float] = {}
    if include_market_prices:
        market_prices = await asyncio.to_thread(_fetch_live_prices, client, ticker)

    preds = model_manager.predict_all(ticker, market_prices=market_prices or None)

    predictions = [
        WordPrediction(
            word=p["word"],
            probability=p.get("probability", 0.0),
            raw_probability=p.get("raw_probability", p.get("probability", 0.0)),
            lower_bound=p.get("lower_bound", 0.0),
            upper_bound=p.get("upper_bound", 1.0),
            uncertainty=p.get("uncertainty", 0.0),
            n_samples=p.get("n_samples", 0),
            market_price=p.get("market_price"),
            edge=p.get("edge"),
            adjusted_edge=p.get("adjusted_edge"),
            trade_signal=p.get("trade_signal", "HOLD"),
            kelly_fraction=p.get("kelly_fraction", 0.0),
            confidence=p.get("confidence"),
        )
        for p in preds
    ]

    return PredictionResponse(
        ticker=ticker,
        predictions=predictions,
        model_type="MarketAdjustedModel",
        generated_at=datetime.now(tz=timezone.utc).isoformat(),
    )


@router.get("/{ticker}/{word}", response_model=WordPrediction)
async def get_prediction_for_word(
    ticker: str,
    word: str,
    client: KalshiClientProtocol = Depends(get_kalshi_client),
    model_manager: ModelManager = Depends(get_model_manager),
    include_market_price: bool = Query(True),
):
    """Get prediction for a single word."""
    ticker = ticker.upper()
    word_key = word.lower()

    if ticker not in model_manager.models:
        raise HTTPException(status_code=404, detail=f"No models for ticker {ticker}")

    if word_key not in model_manager.models.get(ticker, {}):
        raise HTTPException(
            status_code=404,
            detail=f"No model for word '{word_key}' on ticker {ticker}",
        )

    market_price: Optional[float] = None
    if include_market_price:
        prices = await asyncio.to_thread(_fetch_live_prices, client, ticker)
        market_price = prices.get(word_key)

    p = model_manager.predict(ticker, word_key, market_price=market_price)

    return WordPrediction(
        word=word_key,
        probability=p.get("probability", 0.0),
        raw_probability=p.get("raw_probability", p.get("probability", 0.0)),
        lower_bound=p.get("lower_bound", 0.0),
        upper_bound=p.get("upper_bound", 1.0),
        uncertainty=p.get("uncertainty", 0.0),
        n_samples=p.get("n_samples", 0),
        market_price=p.get("market_price"),
        edge=p.get("edge"),
        adjusted_edge=p.get("adjusted_edge"),
        trade_signal=p.get("trade_signal", "HOLD"),
        kelly_fraction=p.get("kelly_fraction", 0.0),
        confidence=p.get("confidence"),
    )
