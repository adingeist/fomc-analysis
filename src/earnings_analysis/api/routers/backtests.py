"""Backtest endpoints."""

from __future__ import annotations

import asyncio

from fastapi import APIRouter, Depends, HTTPException, Query

from earnings_analysis.api.dependencies import get_model_manager
from earnings_analysis.api.schemas import (
    BacktestMetricsSchema,
    BacktestResponse,
    BacktestTradeSchema,
)
from earnings_analysis.api.services.backtest_service import load_backtest, run_backtest
from earnings_analysis.api.services.model_manager import ModelManager

router = APIRouter(prefix="/backtests", tags=["backtests"])


@router.get("/{ticker}", response_model=BacktestResponse)
async def get_backtest(
    ticker: str,
    model_manager: ModelManager = Depends(get_model_manager),
):
    """Retrieve the most recent backtest result for a ticker."""
    ticker = ticker.upper()

    if ticker not in model_manager.models:
        raise HTTPException(
            status_code=404,
            detail=f"No models for ticker {ticker}. Available: {list(model_manager.models.keys())}",
        )

    data = load_backtest(ticker)
    if data is None:
        raise HTTPException(
            status_code=404,
            detail=f"No persisted backtest results for {ticker}. POST to this endpoint to run one.",
        )

    return _build_response(data)


@router.post("/{ticker}", response_model=BacktestResponse)
async def trigger_backtest(
    ticker: str,
    model_manager: ModelManager = Depends(get_model_manager),
    edge_threshold: float = Query(0.12, ge=0.0, le=1.0),
    initial_capital: float = Query(10000.0, gt=0),
):
    """Run a new backtest for a ticker and return results."""
    ticker = ticker.upper()

    if ticker not in model_manager.models:
        raise HTTPException(
            status_code=404,
            detail=f"No models for ticker {ticker}. Available: {list(model_manager.models.keys())}",
        )

    data = await asyncio.to_thread(
        run_backtest, ticker, model_manager, edge_threshold, initial_capital
    )

    return _build_response(data)


def _build_response(data: dict) -> BacktestResponse:
    metrics = BacktestMetricsSchema(**data["metrics"])
    trades = [BacktestTradeSchema(**t) for t in data.get("trades", [])]
    return BacktestResponse(
        ticker=data["ticker"],
        metrics=metrics,
        trades=trades,
        metadata=data.get("metadata", {}),
    )
