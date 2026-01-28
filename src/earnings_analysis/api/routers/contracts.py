"""Contract discovery endpoints."""

from __future__ import annotations

import asyncio
from typing import Optional

from fastapi import APIRouter, Depends, Query

from earnings_analysis.api.dependencies import get_kalshi_client, get_model_manager
from earnings_analysis.api.schemas import ContractSchema, ContractsResponse
from earnings_analysis.api.services.model_manager import KNOWN_TICKERS, ModelManager
from fomc_analysis.kalshi_client_factory import KalshiClientProtocol

router = APIRouter(prefix="/contracts", tags=["contracts"])

_SERIES_PREFIX = "KXEARNINGSMENTION"


def _fetch_contracts(
    client: KalshiClientProtocol, ticker: str, status: Optional[str]
) -> list[dict]:
    series_ticker = f"{_SERIES_PREFIX}{ticker.upper()}"
    if status and status != "all":
        markets = client.get_markets(series_ticker=series_ticker, status=status)
    else:
        markets = client.get_markets(series_ticker=series_ticker)
    return markets


def _extract_word_from_market(market: dict) -> Optional[str]:
    custom_strike = market.get("custom_strike") or {}
    word = custom_strike.get("Word") or market.get("yes_sub_title", "")
    return word.strip() if word else None


@router.get("/{ticker}", response_model=ContractsResponse)
async def get_contracts(
    ticker: str,
    client: KalshiClientProtocol = Depends(get_kalshi_client),
    status: str = Query("active", description="Filter: active, settled, finalized, all"),
):
    """List Kalshi contracts for a ticker."""
    ticker = ticker.upper()
    markets = await asyncio.to_thread(_fetch_contracts, client, ticker, status)

    contracts: list[ContractSchema] = []
    active_count = 0

    for market in markets:
        word = _extract_word_from_market(market)
        if not word:
            continue
        mkt_status = (market.get("status") or "").lower()
        if mkt_status in ("active", "open"):
            active_count += 1
        contracts.append(
            ContractSchema(
                market_ticker=market.get("ticker", ""),
                word=word,
                status=mkt_status,
                last_price=(market.get("last_price") or 0) / 100.0,
                yes_bid=(market.get("yes_bid") or 0) / 100.0 if market.get("yes_bid") else None,
                yes_ask=(market.get("yes_ask") or 0) / 100.0 if market.get("yes_ask") else None,
                expiration_time=market.get("expiration_time"),
            )
        )

    return ContractsResponse(
        ticker=ticker,
        contracts=contracts,
        active_count=active_count,
        total_count=len(contracts),
    )


@router.get("", response_model=list[dict])
async def list_available_tickers(
    client: KalshiClientProtocol = Depends(get_kalshi_client),
    model_manager: ModelManager = Depends(get_model_manager),
):
    """List all known tickers and their contract counts."""
    results = []
    for ticker in KNOWN_TICKERS:
        model_words = list(model_manager.models.get(ticker, {}).keys())
        results.append(
            {
                "ticker": ticker,
                "series_ticker": f"{_SERIES_PREFIX}{ticker}",
                "models_trained": len(model_words) > 0,
                "n_model_words": len(model_words),
            }
        )
    return results
