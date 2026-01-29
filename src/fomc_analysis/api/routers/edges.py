"""FOMC edges API router."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter, Depends, Query, Request

from fomc_analysis.api.schemas import (
    FOMCEdgeOpportunity,
    FOMCEdgesResponse,
)

router = APIRouter(prefix="/edges", tags=["fomc-edges"])


def get_model_service(request: Request):
    """Dependency to get FOMC model service."""
    return request.app.state.fomc_model_service


def get_kalshi_client(request: Request):
    """Dependency to get Kalshi client."""
    return getattr(request.app.state, "kalshi_client", None)


@router.get("", response_model=FOMCEdgesResponse)
async def get_fomc_edges(
    min_edge: float = Query(default=0.05, ge=0.0, le=1.0, description="Minimum edge threshold"),
    signal: Optional[str] = Query(default=None, description="Filter by signal: 'BUY_YES' or 'BUY_NO'"),
    model_service=Depends(get_model_service),
    kalshi_client=Depends(get_kalshi_client),
):
    """
    Find FOMC trading opportunities with positive edge.

    Returns contracts where the predicted probability differs significantly
    from the market price, indicating a potential trading opportunity.
    """
    opportunities = model_service.get_edges(min_edge=min_edge, kalshi_client=kalshi_client)

    if signal:
        signal_upper = signal.upper()
        opportunities = [o for o in opportunities if o.get("signal") == signal_upper]

    total_contracts = model_service.contract_count

    return FOMCEdgesResponse(
        opportunities=[
            FOMCEdgeOpportunity(
                contract=o.get("contract", ""),
                predicted_probability=o.get("predicted_probability", 0.0),
                market_price=o.get("market_price", 0.0),
                edge=o.get("edge", 0.0),
                signal=o.get("signal", "HOLD"),
                confidence_lower=o.get("confidence_lower", 0.0),
                confidence_upper=o.get("confidence_upper", 1.0),
                market_ticker=o.get("market_ticker"),
            )
            for o in opportunities
        ],
        total_contracts_scanned=total_contracts,
        generated_at=datetime.now(timezone.utc).isoformat(),
    )
