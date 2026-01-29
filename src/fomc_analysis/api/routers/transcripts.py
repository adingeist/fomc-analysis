"""FOMC transcripts API router."""

from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Request

from fomc_analysis.api.schemas import (
    TranscriptResponse,
    TranscriptSegment,
    TranscriptSummary,
    TranscriptsListResponse,
)

router = APIRouter(prefix="/transcripts", tags=["fomc-transcripts"])


def get_data_service(request: Request):
    """Dependency to get FOMC data service."""
    return request.app.state.fomc_data_service


@router.get("", response_model=TranscriptsListResponse)
async def list_transcripts(
    limit: int = Query(default=50, ge=1, le=200, description="Max transcripts to return"),
    data_service=Depends(get_data_service),
):
    """
    List available FOMC transcripts with summary information.

    Returns a list of all transcripts with metadata about segments
    and word counts.
    """
    transcripts = data_service.get_available_transcripts()[:limit]

    return TranscriptsListResponse(
        transcripts=[
            TranscriptSummary(
                meeting_date=t.meeting_date,
                total_segments=t.total_segments,
                powell_segments=t.powell_segments,
                word_count=t.word_count,
                available=t.available,
            )
            for t in transcripts
        ],
        total_transcripts=len(data_service.get_available_transcripts()),
    )


@router.get("/{meeting_date}", response_model=TranscriptResponse)
async def get_transcript(
    meeting_date: str,
    powell_only: bool = Query(default=False, description="Return only Powell's segments"),
    data_service=Depends(get_data_service),
):
    """
    Get full transcript for a specific FOMC meeting.

    Parameters
    ----------
    meeting_date : str
        Meeting date in YYYY-MM-DD or YYYYMMDD format.
    powell_only : bool
        If True, return only Powell's speaking segments.
    """
    normalized_date = meeting_date.replace("-", "")
    if len(normalized_date) == 8:
        meeting_date = f"{normalized_date[:4]}-{normalized_date[4:6]}-{normalized_date[6:8]}"

    result = data_service.get_transcript(meeting_date)

    if result is None:
        raise HTTPException(
            status_code=404,
            detail=f"Transcript not found for meeting date: {meeting_date}",
        )

    segments = result.get("segments", [])

    if powell_only:
        segments = [s for s in segments if s.get("role") == "powell"]

    return TranscriptResponse(
        meeting_date=result.get("meeting_date", meeting_date),
        segments=[
            TranscriptSegment(
                segment_idx=s.get("segment_idx", 0),
                speaker=s.get("speaker", "Unknown"),
                role=s.get("role", "unknown"),
                text=s.get("text", ""),
            )
            for s in segments
        ],
        total_segments=len(segments),
        powell_word_count=result.get("powell_word_count", 0),
    )
