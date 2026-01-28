"""Earnings word frequencies API router."""

from __future__ import annotations

from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Request

from earnings_analysis.api.schemas import (
    EarningsWordFrequenciesResponse,
    EarningsWordFrequencyRecord,
    EarningsWordFrequencySeries,
)

router = APIRouter(prefix="/word-frequencies", tags=["earnings-word-frequencies"])


def get_model_manager(request: Request):
    """Dependency to get model manager."""
    return request.app.state.model_manager


@router.get("/{ticker}", response_model=EarningsWordFrequenciesResponse)
async def get_word_frequencies(
    ticker: str,
    words: Optional[List[str]] = Query(default=None, description="Filter to specific words"),
    limit: int = Query(default=50, ge=1, le=200, description="Max calls to return"),
    model_manager=Depends(get_model_manager),
):
    """
    Get historical word frequency data across earnings calls for a ticker.

    Returns counts of how many times each tracked word was mentioned
    in executive remarks for each earnings call.
    """
    ticker_upper = ticker.upper()

    if ticker_upper not in model_manager.models:
        raise HTTPException(
            status_code=404,
            detail=f"Ticker '{ticker}' not found. Available: {list(model_manager.models.keys())}",
        )

    # Get ground truth data which contains historical word mentions
    ground_truth = model_manager.ground_truth.get(ticker_upper)
    if ground_truth is None:
        return EarningsWordFrequenciesResponse(
            ticker=ticker_upper,
            words=[],
            call_dates=[],
            total_calls=0,
        )

    # Build word frequency data from ground truth
    outcomes_df = ground_truth.outcomes_df
    if outcomes_df is None or outcomes_df.empty:
        return EarningsWordFrequenciesResponse(
            ticker=ticker_upper,
            words=[],
            call_dates=[],
            total_calls=0,
        )

    # Filter to requested words if specified
    available_words = list(outcomes_df.columns)
    if words:
        available_words = [w for w in words if w in outcomes_df.columns]

    word_series = []
    for word in available_words:
        series = outcomes_df[word]
        frequencies = []

        for call_date, mentioned in series.items():
            frequencies.append(EarningsWordFrequencyRecord(
                call_date=str(call_date),
                word=word,
                count=1 if mentioned else 0,  # Binary for earnings
                mentioned=bool(mentioned),
            ))

        # Limit results
        frequencies = frequencies[-limit:]

        total_mentions = int(series.sum()) if series.notna().any() else 0
        mention_rate = float(series.mean()) if len(series) > 0 else 0.0

        word_series.append(EarningsWordFrequencySeries(
            word=word,
            frequencies=frequencies,
            total_mentions=total_mentions,
            mention_rate=mention_rate,
        ))

    call_dates = [str(d) for d in outcomes_df.index.tolist()][-limit:]

    return EarningsWordFrequenciesResponse(
        ticker=ticker_upper,
        words=word_series,
        call_dates=call_dates,
        total_calls=len(outcomes_df),
    )


@router.get("/{ticker}/{word}", response_model=EarningsWordFrequencySeries)
async def get_word_frequency(
    ticker: str,
    word: str,
    limit: int = Query(default=50, ge=1, le=200, description="Max calls to return"),
    model_manager=Depends(get_model_manager),
):
    """
    Get frequency history for a specific word for a ticker.

    Parameters
    ----------
    ticker : str
        Company ticker (e.g., META, TSLA, NVDA).
    word : str
        The word to get frequency history for.
    """
    ticker_upper = ticker.upper()

    if ticker_upper not in model_manager.models:
        raise HTTPException(
            status_code=404,
            detail=f"Ticker '{ticker}' not found. Available: {list(model_manager.models.keys())}",
        )

    ground_truth = model_manager.ground_truth.get(ticker_upper)
    if ground_truth is None:
        raise HTTPException(
            status_code=404,
            detail=f"No ground truth data for ticker '{ticker}'",
        )

    outcomes_df = ground_truth.outcomes_df
    if outcomes_df is None or word not in outcomes_df.columns:
        raise HTTPException(
            status_code=404,
            detail=f"Word '{word}' not found for ticker '{ticker}'",
        )

    series = outcomes_df[word]
    frequencies = []

    for call_date, mentioned in series.items():
        frequencies.append(EarningsWordFrequencyRecord(
            call_date=str(call_date),
            word=word,
            count=1 if mentioned else 0,
            mentioned=bool(mentioned),
        ))

    frequencies = frequencies[-limit:]

    total_mentions = int(series.sum()) if series.notna().any() else 0
    mention_rate = float(series.mean()) if len(series) > 0 else 0.0

    return EarningsWordFrequencySeries(
        word=word,
        frequencies=frequencies,
        total_mentions=total_mentions,
        mention_rate=mention_rate,
    )
