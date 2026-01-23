"""Configuration management for earnings analysis."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field


class EarningsConfig(BaseModel):
    """Configuration for earnings analysis."""

    # Data directories
    data_dir: Path = Field(default_factory=lambda: Path("data/earnings"))
    transcripts_dir: Path = Field(default_factory=lambda: Path("data/earnings/transcripts"))
    segments_dir: Path = Field(default_factory=lambda: Path("data/earnings/segments"))
    features_dir: Path = Field(default_factory=lambda: Path("data/earnings/features"))
    outcomes_dir: Path = Field(default_factory=lambda: Path("data/earnings/outcomes"))
    keywords_dir: Path = Field(default_factory=lambda: Path("data/earnings/keywords"))

    # API keys
    alpha_vantage_api_key: Optional[str] = Field(
        default_factory=lambda: os.getenv("ALPHA_VANTAGE_API_KEY")
    )
    openai_api_key: Optional[str] = Field(
        default_factory=lambda: os.getenv("OPENAI_API_KEY")
    )

    # Database
    database_url: str = Field(
        default_factory=lambda: os.getenv(
            "DATABASE_URL",
            "sqlite:///data/fomc_analysis.db"  # Reuse existing DB
        )
    )

    # Parsing defaults
    use_ai_segmentation: bool = False
    openai_model: str = "gpt-4o-mini"

    # Feature extraction defaults
    speaker_mode: str = "executives_only"  # executives_only, full_transcript
    phrase_mode: str = "variants"  # strict_literal, variants

    # Backtesting defaults
    initial_capital: float = 10000.0
    position_size_pct: float = 0.05  # 5% per trade (higher than FOMC)

    class Config:
        """Pydantic config."""
        arbitrary_types_allowed = True


def get_config() -> EarningsConfig:
    """Get the global earnings analysis configuration."""
    return EarningsConfig()
