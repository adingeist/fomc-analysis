"""FOMC contracts API router."""

from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, Depends, Query, Request

from fomc_analysis.api.schemas import (
    FOMCContractSchema,
    FOMCContractsResponse,
)

router = APIRouter(prefix="/contracts", tags=["fomc-contracts"])


def get_model_service(request: Request):
    """Dependency to get FOMC model service."""
    return request.app.state.fomc_model_service


def get_kalshi_client(request: Request):
    """Dependency to get Kalshi client."""
    return getattr(request.app.state, "kalshi_client", None)


@router.get("", response_model=FOMCContractsResponse)
async def get_fomc_contracts(
    status: Optional[str] = Query(
        default=None,
        description="Filter by status: 'active', 'settled', or 'all'",
    ),
    model_service=Depends(get_model_service),
    kalshi_client=Depends(get_kalshi_client),
):
    """
    Get FOMC mention contracts from Kalshi.

    Returns all tracked FOMC word/phrase contracts with their current
    market prices and status.
    """
    contracts = model_service.get_contracts(kalshi_client, status=status)

    active_count = sum(
        1 for c in contracts
        if c.get("status") in {"open", "active"}
    )
    settled_count = sum(
        1 for c in contracts
        if c.get("status") in {"settled", "resolved"}
    )

    return FOMCContractsResponse(
        contracts=[
            FOMCContractSchema(
                market_ticker=c.get("market_ticker", ""),
                word=c.get("word", ""),
                threshold=c.get("threshold", 1),
                status=c.get("status", "unknown"),
                last_price=c.get("last_price"),
                yes_bid=c.get("yes_bid"),
                yes_ask=c.get("yes_ask"),
                expiration_time=c.get("expiration_time"),
                result=c.get("result"),
            )
            for c in contracts
        ],
        active_count=active_count,
        settled_count=settled_count,
        total_count=len(contracts),
    )
