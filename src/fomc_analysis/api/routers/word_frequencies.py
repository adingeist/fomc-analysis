"""FOMC word frequencies API router."""

from __future__ import annotations

from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Request

from fomc_analysis.api.schemas import (
    WordFrequenciesResponse,
    WordFrequencyRecord,
    WordFrequencySeries,
)

router = APIRouter(prefix="/word-frequencies", tags=["fomc-word-frequencies"])


def get_data_service(request: Request):
    """Dependency to get FOMC data service."""
    return request.app.state.fomc_data_service


@router.get("", response_model=WordFrequenciesResponse)
async def get_word_frequencies(
    words: Optional[List[str]] = Query(default=None, description="Filter to specific words"),
    limit: int = Query(default=50, ge=1, le=200, description="Max meetings to return"),
    data_service=Depends(get_data_service),
):
    """
    Get historical word frequency data across FOMC meetings.

    Returns counts of how many times each tracked word/phrase was mentioned
    in Powell's remarks for each meeting.
    """
    result = data_service.get_word_frequencies(words=words)

    word_series = []
    for word_data in result.get("words", []):
        frequencies = word_data.get("frequencies", [])[-limit:]
        word_series.append(WordFrequencySeries(
            word=word_data.get("word", ""),
            frequencies=[
                WordFrequencyRecord(
                    meeting_date=f.get("meeting_date", ""),
                    word=f.get("word", ""),
                    count=f.get("count", 0),
                    mentioned=f.get("mentioned", False),
                )
                for f in frequencies
            ],
            total_mentions=word_data.get("total_mentions", 0),
            mention_rate=word_data.get("mention_rate", 0.0),
        ))

    meeting_dates = result.get("meeting_dates", [])[-limit:]

    return WordFrequenciesResponse(
        words=word_series,
        meeting_dates=meeting_dates,
        total_meetings=result.get("total_meetings", 0),
    )


@router.get("/{word}", response_model=WordFrequencySeries)
async def get_word_frequency(
    word: str,
    limit: int = Query(default=50, ge=1, le=200, description="Max meetings to return"),
    data_service=Depends(get_data_service),
):
    """
    Get frequency history for a specific word/phrase.

    Parameters
    ----------
    word : str
        The word or phrase to get frequency history for.
    """
    result = data_service.get_word_frequencies(words=[word])

    word_data_list = result.get("words", [])
    if not word_data_list:
        raise HTTPException(
            status_code=404,
            detail=f"Word '{word}' not found in tracked contracts",
        )

    word_data = word_data_list[0]
    frequencies = word_data.get("frequencies", [])[-limit:]

    return WordFrequencySeries(
        word=word_data.get("word", word),
        frequencies=[
            WordFrequencyRecord(
                meeting_date=f.get("meeting_date", ""),
                word=f.get("word", ""),
                count=f.get("count", 0),
                mentioned=f.get("mentioned", False),
            )
            for f in frequencies
        ],
        total_mentions=word_data.get("total_mentions", 0),
        mention_rate=word_data.get("mention_rate", 0.0),
    )
