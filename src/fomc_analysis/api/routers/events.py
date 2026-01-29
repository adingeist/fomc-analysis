"""FOMC upcoming events API router."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter, Depends, Request
from pydantic import BaseModel

router = APIRouter(prefix="/events", tags=["fomc-events"])

# 2026 FOMC meeting dates (Fed publishes these annually)
FOMC_MEETING_DATES_2026 = [
    "2026-01-28",  # Jan 28-29
    "2026-03-18",  # Mar 18-19
    "2026-05-06",  # May 6-7
    "2026-06-17",  # Jun 17-18
    "2026-07-29",  # Jul 29-30
    "2026-09-16",  # Sep 16-17
    "2026-11-04",  # Nov 4-5
    "2026-12-16",  # Dec 16-17
]


class FOMCEvent(BaseModel):
    meeting_date: str
    days_until: int
    has_projections: bool
    status: str  # "upcoming", "today", "past"


class FOMCEventsResponse(BaseModel):
    next_meeting: Optional[FOMCEvent]
    upcoming_meetings: list[FOMCEvent]
    generated_at: str


def get_model_service(request: Request):
    """Dependency to get FOMC model service."""
    return request.app.state.fomc_model_service


@router.get("", response_model=FOMCEventsResponse)
async def get_fomc_events(
    model_service=Depends(get_model_service),
):
    """
    Get upcoming FOMC meeting dates.

    Returns the next meeting date and list of all upcoming meetings
    with days until each meeting.
    """
    today = datetime.now(timezone.utc).date()
    today_str = today.isoformat()

    # Meetings with SEP projections (typically Mar, Jun, Sep, Dec)
    projection_months = {3, 6, 9, 12}

    upcoming = []
    next_meeting = None

    for date_str in FOMC_MEETING_DATES_2026:
        meeting_date = datetime.strptime(date_str, "%Y-%m-%d").date()
        days_until = (meeting_date - today).days
        month = meeting_date.month

        if days_until < 0:
            status = "past"
        elif days_until == 0:
            status = "today"
        else:
            status = "upcoming"

        event = FOMCEvent(
            meeting_date=date_str,
            days_until=days_until,
            has_projections=month in projection_months,
            status=status,
        )

        if status != "past":
            upcoming.append(event)
            if next_meeting is None:
                next_meeting = event

    return FOMCEventsResponse(
        next_meeting=next_meeting,
        upcoming_meetings=upcoming,
        generated_at=datetime.now(timezone.utc).isoformat(),
    )


@router.get("/next", response_model=FOMCEvent)
async def get_next_fomc_meeting(
    model_service=Depends(get_model_service),
):
    """
    Get the next FOMC meeting date.

    Returns details about the upcoming meeting including days until
    and whether it includes economic projections.
    """
    today = datetime.now(timezone.utc).date()
    projection_months = {3, 6, 9, 12}

    for date_str in FOMC_MEETING_DATES_2026:
        meeting_date = datetime.strptime(date_str, "%Y-%m-%d").date()
        days_until = (meeting_date - today).days

        if days_until >= 0:
            return FOMCEvent(
                meeting_date=date_str,
                days_until=days_until,
                has_projections=meeting_date.month in projection_months,
                status="today" if days_until == 0 else "upcoming",
            )

    # Fallback if no future meetings in list
    return FOMCEvent(
        meeting_date="2027-01-27",
        days_until=365,
        has_projections=False,
        status="upcoming",
    )
