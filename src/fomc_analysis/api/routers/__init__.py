"""FOMC API routers package."""

from .predictions import router as predictions_router
from .word_frequencies import router as word_frequencies_router
from .transcripts import router as transcripts_router
from .backtests import router as backtests_router
from .contracts import router as contracts_router
from .edges import router as edges_router

__all__ = [
    "predictions_router",
    "word_frequencies_router",
    "transcripts_router",
    "backtests_router",
    "contracts_router",
    "edges_router",
]
