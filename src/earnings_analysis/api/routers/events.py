"""Earnings upcoming events API router."""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter, Depends, Query, Request
from pydantic import BaseModel

router = APIRouter(prefix="/events", tags=["earnings-events"])


class EarningsEvent(BaseModel):
    ticker: str
    company_name: str
    earnings_date: Optional[str]
    days_until: Optional[int]
    time_of_day: str  # "BMO" (before market open), "AMC" (after market close), "TBD"
    status: str  # "upcoming", "today", "past", "unknown"
    fiscal_quarter: Optional[str]


class EarningsEventsResponse(BaseModel):
    events: list[EarningsEvent]
    generated_at: str


class NextEarningsResponse(BaseModel):
    ticker: str
    company_name: str
    earnings_date: Optional[str]
    days_until: Optional[int]
    time_of_day: str
    status: str
    fiscal_quarter: Optional[str]
    contracts_available: int


# Company metadata
COMPANY_INFO = {
    "META": {"name": "Meta Platforms, Inc.", "typical_time": "AMC"},
    "TSLA": {"name": "Tesla, Inc.", "typical_time": "AMC"},
    "NVDA": {"name": "NVIDIA Corporation", "typical_time": "AMC"},
    "AAPL": {"name": "Apple Inc.", "typical_time": "AMC"},
    "GOOGL": {"name": "Alphabet Inc.", "typical_time": "AMC"},
    "MSFT": {"name": "Microsoft Corporation", "typical_time": "AMC"},
    "AMZN": {"name": "Amazon.com, Inc.", "typical_time": "AMC"},
}


def get_model_manager(request: Request):
    """Dependency to get model manager."""
    return request.app.state.model_manager


def get_kalshi_client(request: Request):
    """Dependency to get Kalshi client."""
    return getattr(request.app.state, "kalshi_client", None)


def _get_next_earnings_from_kalshi(client, ticker: str) -> Optional[dict]:
    """Try to get next earnings date from Kalshi contract expiration."""
    if client is None:
        return None

    try:
        series_ticker = f"KXEARNINGSMENTION{ticker.upper()}"
        markets = client.get_markets(series_ticker=series_ticker, status="open")

        if not markets:
            return None

        # Find earliest expiration among active contracts
        earliest_exp = None
        for market in markets:
            exp_time = market.get("expiration_time")
            if exp_time:
                try:
                    exp_dt = datetime.fromisoformat(exp_time.replace("Z", "+00:00"))
                    if earliest_exp is None or exp_dt < earliest_exp:
                        earliest_exp = exp_dt
                except (ValueError, TypeError):
                    continue

        if earliest_exp:
            return {
                "date": earliest_exp.date().isoformat(),
                "datetime": earliest_exp,
            }
    except Exception:
        pass

    return None


@router.get("", response_model=EarningsEventsResponse)
async def get_earnings_events(
    model_manager=Depends(get_model_manager),
    kalshi_client=Depends(get_kalshi_client),
):
    """
    Get upcoming earnings dates for all tracked tickers.

    Returns earnings dates derived from Kalshi contract expirations
    when available.
    """
    today = datetime.now(timezone.utc).date()
    events = []

    for ticker in model_manager.models.keys():
        info = COMPANY_INFO.get(ticker, {"name": ticker, "typical_time": "TBD"})

        # Try to get earnings date from Kalshi
        earnings_info = await asyncio.to_thread(
            _get_next_earnings_from_kalshi, kalshi_client, ticker
        )

        if earnings_info:
            earnings_date = earnings_info["date"]
            earnings_dt = datetime.strptime(earnings_date, "%Y-%m-%d").date()
            days_until = (earnings_dt - today).days

            if days_until < 0:
                status = "past"
            elif days_until == 0:
                status = "today"
            else:
                status = "upcoming"

            events.append(EarningsEvent(
                ticker=ticker,
                company_name=info["name"],
                earnings_date=earnings_date,
                days_until=days_until,
                time_of_day=info["typical_time"],
                status=status,
                fiscal_quarter=None,
            ))
        else:
            events.append(EarningsEvent(
                ticker=ticker,
                company_name=info["name"],
                earnings_date=None,
                days_until=None,
                time_of_day=info["typical_time"],
                status="unknown",
                fiscal_quarter=None,
            ))

    # Sort by days_until (unknown at end)
    events.sort(key=lambda e: (e.days_until is None, e.days_until or 999))

    return EarningsEventsResponse(
        events=events,
        generated_at=datetime.now(timezone.utc).isoformat(),
    )


@router.get("/{ticker}", response_model=NextEarningsResponse)
async def get_next_earnings(
    ticker: str,
    model_manager=Depends(get_model_manager),
    kalshi_client=Depends(get_kalshi_client),
):
    """
    Get next earnings date for a specific ticker.

    Returns the upcoming earnings call date derived from Kalshi
    contract expirations.
    """
    ticker = ticker.upper()
    today = datetime.now(timezone.utc).date()
    info = COMPANY_INFO.get(ticker, {"name": ticker, "typical_time": "TBD"})

    # Get earnings date from Kalshi
    earnings_info = await asyncio.to_thread(
        _get_next_earnings_from_kalshi, kalshi_client, ticker
    )

    # Count available contracts
    contracts_count = 0
    if kalshi_client:
        try:
            series_ticker = f"KXEARNINGSMENTION{ticker}"
            markets = kalshi_client.get_markets(series_ticker=series_ticker, status="open")
            contracts_count = len(markets) if markets else 0
        except Exception:
            pass

    if earnings_info:
        earnings_date = earnings_info["date"]
        earnings_dt = datetime.strptime(earnings_date, "%Y-%m-%d").date()
        days_until = (earnings_dt - today).days

        if days_until < 0:
            status = "past"
        elif days_until == 0:
            status = "today"
        else:
            status = "upcoming"

        return NextEarningsResponse(
            ticker=ticker,
            company_name=info["name"],
            earnings_date=earnings_date,
            days_until=days_until,
            time_of_day=info["typical_time"],
            status=status,
            fiscal_quarter=None,
            contracts_available=contracts_count,
        )

    return NextEarningsResponse(
        ticker=ticker,
        company_name=info["name"],
        earnings_date=None,
        days_until=None,
        time_of_day=info["typical_time"],
        status="unknown",
        fiscal_quarter=None,
        contracts_available=contracts_count,
    )
