"""FOMC predictions API router."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Request

from fomc_analysis.api.schemas import (
    FOMCPredictionResponse,
    FOMCWordPrediction,
)

router = APIRouter(prefix="/predictions", tags=["fomc-predictions"])


def get_model_service(request: Request):
    """Dependency to get FOMC model service."""
    return request.app.state.fomc_model_service


def get_kalshi_client(request: Request):
    """Dependency to get Kalshi client."""
    return getattr(request.app.state, "kalshi_client", None)


@router.get("", response_model=FOMCPredictionResponse)
async def get_fomc_predictions(
    include_market_prices: bool = True,
    model_service=Depends(get_model_service),
    kalshi_client=Depends(get_kalshi_client),
):
    """
    Get FOMC word mention predictions for the next meeting.

    Returns predicted probabilities for each tracked word/phrase contract,
    along with market prices and trading edges when available.
    """
    result = model_service.get_predictions(kalshi_client if include_market_prices else None)

    predictions = []
    for pred in result.get("predictions", []):
        edge = pred.get("edge")
        market_price = pred.get("market_price")

        if edge is not None and edge > 0.1:
            signal = "BUY_YES"
        elif edge is not None and edge < -0.1:
            signal = "BUY_NO"
        else:
            signal = "HOLD"

        predictions.append(FOMCWordPrediction(
            contract=pred.get("contract", ""),
            probability=pred.get("predicted_probability", 0.0),
            lower_bound=pred.get("confidence_lower", 0.0),
            upper_bound=pred.get("confidence_upper", 1.0),
            uncertainty=(pred.get("confidence_upper", 1.0) - pred.get("confidence_lower", 0.0)) / 2,
            market_price=market_price,
            edge=edge,
            trade_signal=signal,
            ticker=pred.get("ticker"),
            event_ticker=pred.get("event_ticker"),
        ))

    predictions.sort(key=lambda p: abs(p.edge or 0), reverse=True)

    metadata = result.get("metadata", {})
    next_meeting = None
    if result.get("predictions"):
        first_pred = result["predictions"][0]
        next_meeting = first_pred.get("meeting_date")

    return FOMCPredictionResponse(
        predictions=predictions,
        next_meeting_date=next_meeting,
        model_type=metadata.get("model_class", "BetaBinomialModel"),
        model_params=metadata.get("model_params", {}),
        training_meetings=metadata.get("total_training_meetings", 0),
        generated_at=metadata.get("generated_at", datetime.now(timezone.utc).isoformat()),
    )


@router.get("/{word}", response_model=FOMCWordPrediction)
async def get_fomc_prediction_by_word(
    word: str,
    include_market_price: bool = True,
    model_service=Depends(get_model_service),
    kalshi_client=Depends(get_kalshi_client),
):
    """
    Get prediction for a specific FOMC word/phrase.

    Parameters
    ----------
    word : str
        The word or phrase to get prediction for (case-insensitive).
    """
    result = model_service.get_predictions(kalshi_client if include_market_price else None)

    word_lower = word.lower()
    for pred in result.get("predictions", []):
        contract = pred.get("contract", "")
        if contract.lower() == word_lower or word_lower in contract.lower():
            edge = pred.get("edge")
            market_price = pred.get("market_price")

            if edge is not None and edge > 0.1:
                signal = "BUY_YES"
            elif edge is not None and edge < -0.1:
                signal = "BUY_NO"
            else:
                signal = "HOLD"

            return FOMCWordPrediction(
                contract=contract,
                probability=pred.get("predicted_probability", 0.0),
                lower_bound=pred.get("confidence_lower", 0.0),
                upper_bound=pred.get("confidence_upper", 1.0),
                uncertainty=(pred.get("confidence_upper", 1.0) - pred.get("confidence_lower", 0.0)) / 2,
                market_price=market_price,
                edge=edge,
                trade_signal=signal,
                ticker=pred.get("ticker"),
                event_ticker=pred.get("event_ticker"),
            )

    raise HTTPException(status_code=404, detail=f"Word '{word}' not found in tracked contracts")
