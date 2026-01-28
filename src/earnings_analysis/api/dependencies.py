"""FastAPI dependency injection providers."""

from __future__ import annotations

from fastapi import Request

from fomc_analysis.kalshi_client_factory import KalshiClientProtocol

from .services.model_manager import ModelManager


def get_kalshi_client(request: Request) -> KalshiClientProtocol:
    """Retrieve the shared Kalshi client from app state."""
    return request.app.state.kalshi_client


def get_model_manager(request: Request) -> ModelManager:
    """Retrieve the shared ModelManager from app state."""
    return request.app.state.model_manager
