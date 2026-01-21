"""
Configuration management using Pydantic Settings.

This module provides a centralized configuration system that loads
environment variables from a .env file or the system environment.
All configuration values are validated using Pydantic.
"""

from __future__ import annotations

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables.

    Settings are loaded from:
    1. Environment variables
    2. .env file in the project root (if present)

    All settings are optional by default, but you should provide
    KALSHI_API_KEY and KALSHI_API_SECRET if you plan to use the
    Kalshi API client.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Kalshi API credentials
    kalshi_api_key: str | None = Field(
        default=None,
        description="Kalshi API key for authentication",
        alias="KALSHI_API_KEY",
    )
    kalshi_api_secret: str | None = Field(
        default=None,
        description="Kalshi API secret for authentication",
        alias="KALSHI_API_SECRET",
    )

    # Optional: Kalshi API base URL (for testing or custom endpoints)
    kalshi_base_url: str = Field(
        default="https://trading-api.kalshi.com/trade-api/v2",
        description="Base URL for the Kalshi API",
        alias="KALSHI_BASE_URL",
    )


# Global settings instance
# Access via: from config import settings
settings = Settings()
