"""Edge-finding endpoints — compare model predictions to live market prices."""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query

from earnings_analysis.api.dependencies import get_kalshi_client, get_model_manager
from earnings_analysis.api.schemas import EdgeOpportunity, EdgesResponse
from earnings_analysis.api.services.edge_finder import (
    find_edges_all_tickers,
    find_edges_for_ticker,
)
from earnings_analysis.api.services.model_manager import ModelManager
from fomc_analysis.kalshi_client_factory import KalshiClientProtocol

router = APIRouter(prefix="/edges", tags=["edges"])


@router.get("/{ticker}", response_model=EdgesResponse)
async def get_edges_for_ticker(
    ticker: str,
    client: KalshiClientProtocol = Depends(get_kalshi_client),
    model_manager: ModelManager = Depends(get_model_manager),
    min_edge: float = Query(0.05, ge=0.0, le=1.0, description="Minimum edge to return"),
    signal: Optional[str] = Query(
        None, description="Filter by signal: BUY_YES or BUY_NO"
    ),
):
    """Find trading edges for a single ticker."""
    ticker = ticker.upper()

    if ticker not in model_manager.models:
        raise HTTPException(
            status_code=404,
            detail=f"No models for ticker {ticker}. Available: {list(model_manager.models.keys())}",
        )

    result = await asyncio.to_thread(
        find_edges_for_ticker, ticker, client, model_manager, min_edge, signal
    )

    opportunities = [EdgeOpportunity(**o) for o in result["opportunities"]]

    return EdgesResponse(
        ticker=ticker,
        opportunities=opportunities,
        total_contracts_scanned=result["total_scanned"],
        generated_at=datetime.now(tz=timezone.utc).isoformat(),
    )


@router.get("", response_model=EdgesResponse)
async def get_edges_all_tickers(
    client: KalshiClientProtocol = Depends(get_kalshi_client),
    model_manager: ModelManager = Depends(get_model_manager),
    min_edge: float = Query(0.05, ge=0.0, le=1.0),
    signal: Optional[str] = Query(None),
):
    """Find trading edges across all tickers with trained models."""
    result = await asyncio.to_thread(
        find_edges_all_tickers, client, model_manager, min_edge, signal
    )

    opportunities = [EdgeOpportunity(**o) for o in result["opportunities"]]

    return EdgesResponse(
        ticker=None,
        opportunities=opportunities,
        total_contracts_scanned=result["total_scanned"],
        generated_at=datetime.now(tz=timezone.utc).isoformat(),
    )
