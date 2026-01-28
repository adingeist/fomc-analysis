"""Earnings transcripts API router."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Request

from earnings_analysis.api.schemas import (
    EarningsTranscriptResponse,
    EarningsTranscriptSegment,
    EarningsTranscriptSummary,
    EarningsTranscriptsListResponse,
)

router = APIRouter(prefix="/transcripts", tags=["earnings-transcripts"])

# Default transcripts directory
TRANSCRIPTS_DIR = Path("data/earnings_transcripts")


def get_model_manager(request: Request):
    """Dependency to get model manager."""
    return request.app.state.model_manager


def _load_transcript_segments(ticker: str, call_date: str) -> Optional[list]:
    """Load transcript segments from JSONL file."""
    # Try multiple path patterns
    patterns = [
        TRANSCRIPTS_DIR / ticker.upper() / f"{call_date.replace('-', '')}.jsonl",
        TRANSCRIPTS_DIR / ticker.lower() / f"{call_date.replace('-', '')}.jsonl",
        TRANSCRIPTS_DIR / f"{ticker.upper()}_{call_date.replace('-', '')}.jsonl",
    ]

    for path in patterns:
        if path.exists():
            segments = []
            with open(path) as f:
                for line in f:
                    if line.strip():
                        segments.append(json.loads(line))
            return segments

    return None


def _get_available_transcripts(ticker: str) -> list:
    """Get list of available transcript files for a ticker."""
    transcripts = []

    # Check ticker-specific directory
    ticker_dir = TRANSCRIPTS_DIR / ticker.upper()
    if ticker_dir.exists():
        for f in sorted(ticker_dir.glob("*.jsonl")):
            date_str = f.stem
            if len(date_str) == 8:
                call_date = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}"
                transcripts.append({"file": f, "call_date": call_date})

    # Also check flat structure
    for f in sorted(TRANSCRIPTS_DIR.glob(f"{ticker.upper()}_*.jsonl")):
        date_str = f.stem.split("_")[1] if "_" in f.stem else ""
        if len(date_str) == 8:
            call_date = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}"
            transcripts.append({"file": f, "call_date": call_date})

    return transcripts


@router.get("/{ticker}", response_model=EarningsTranscriptsListResponse)
async def list_transcripts(
    ticker: str,
    limit: int = Query(default=50, ge=1, le=200, description="Max transcripts to return"),
    model_manager=Depends(get_model_manager),
):
    """
    List available earnings call transcripts for a ticker.

    Returns a list of all transcripts with metadata about segments
    and word counts.
    """
    ticker_upper = ticker.upper()

    available = _get_available_transcripts(ticker_upper)

    summaries = []
    for item in available[-limit:]:
        try:
            segments = _load_transcript_segments(ticker_upper, item["call_date"])
            if segments:
                exec_segments = [
                    s for s in segments
                    if s.get("role") in {"ceo", "cfo", "executive"}
                ]
                word_count = sum(
                    len(s.get("text", "").split()) for s in exec_segments
                )
                summaries.append(EarningsTranscriptSummary(
                    ticker=ticker_upper,
                    call_date=item["call_date"],
                    total_segments=len(segments),
                    executive_segments=len(exec_segments),
                    word_count=word_count,
                    available=True,
                ))
            else:
                summaries.append(EarningsTranscriptSummary(
                    ticker=ticker_upper,
                    call_date=item["call_date"],
                    total_segments=0,
                    executive_segments=0,
                    word_count=0,
                    available=False,
                ))
        except Exception:
            summaries.append(EarningsTranscriptSummary(
                ticker=ticker_upper,
                call_date=item["call_date"],
                total_segments=0,
                executive_segments=0,
                word_count=0,
                available=False,
            ))

    # Sort by date descending
    summaries.sort(key=lambda x: x.call_date, reverse=True)

    return EarningsTranscriptsListResponse(
        ticker=ticker_upper,
        transcripts=summaries,
        total_transcripts=len(available),
    )


@router.get("/{ticker}/{call_date}", response_model=EarningsTranscriptResponse)
async def get_transcript(
    ticker: str,
    call_date: str,
    executives_only: bool = Query(default=False, description="Return only executive segments"),
    model_manager=Depends(get_model_manager),
):
    """
    Get full transcript for a specific earnings call.

    Parameters
    ----------
    ticker : str
        Company ticker (e.g., META, TSLA, NVDA).
    call_date : str
        Call date in YYYY-MM-DD or YYYYMMDD format.
    executives_only : bool
        If True, return only CEO/CFO speaking segments.
    """
    ticker_upper = ticker.upper()

    # Normalize date format
    normalized_date = call_date.replace("-", "")
    if len(normalized_date) == 8:
        call_date = f"{normalized_date[:4]}-{normalized_date[4:6]}-{normalized_date[6:8]}"

    segments = _load_transcript_segments(ticker_upper, call_date)

    if segments is None:
        raise HTTPException(
            status_code=404,
            detail=f"Transcript not found for {ticker_upper} on {call_date}",
        )

    if executives_only:
        segments = [
            s for s in segments
            if s.get("role") in {"ceo", "cfo", "executive"}
        ]

    exec_segments = [
        s for s in segments
        if s.get("role") in {"ceo", "cfo", "executive"}
    ]
    exec_word_count = sum(len(s.get("text", "").split()) for s in exec_segments)

    return EarningsTranscriptResponse(
        ticker=ticker_upper,
        call_date=call_date,
        segments=[
            EarningsTranscriptSegment(
                segment_idx=i,
                speaker=s.get("speaker", "Unknown"),
                role=s.get("role", "unknown"),
                text=s.get("text", ""),
            )
            for i, s in enumerate(segments)
        ],
        total_segments=len(segments),
        executive_word_count=exec_word_count,
    )
