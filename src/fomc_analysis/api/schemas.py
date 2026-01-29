"""Pydantic request/response schemas for the FOMC API."""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel


# --- Predictions ---


class FOMCWordPrediction(BaseModel):
    """Prediction for a single FOMC word/phrase contract."""

    contract: str
    probability: float
    lower_bound: float
    upper_bound: float
    uncertainty: float
    market_price: Optional[float] = None
    edge: Optional[float] = None
    trade_signal: str = "HOLD"
    ticker: Optional[str] = None
    event_ticker: Optional[str] = None


class FOMCPredictionResponse(BaseModel):
    """Response containing all FOMC predictions for the next meeting."""

    predictions: list[FOMCWordPrediction]
    next_meeting_date: Optional[str] = None
    model_type: str
    model_params: dict
    training_meetings: int
    generated_at: str


# --- Word Frequencies ---


class WordFrequencyRecord(BaseModel):
    """Word frequency for a single meeting."""

    meeting_date: str
    word: str
    count: int
    mentioned: bool


class WordFrequencySeries(BaseModel):
    """Time series of word frequencies for a single word."""

    word: str
    frequencies: list[WordFrequencyRecord]
    total_mentions: int
    mention_rate: float


class WordFrequenciesResponse(BaseModel):
    """Response containing word frequency data."""

    words: list[WordFrequencySeries]
    meeting_dates: list[str]
    total_meetings: int


# --- Transcripts ---


class TranscriptSegment(BaseModel):
    """A single segment from a transcript."""

    segment_idx: int
    speaker: str
    role: str
    text: str


class TranscriptSummary(BaseModel):
    """Summary of a single transcript."""

    meeting_date: str
    total_segments: int
    powell_segments: int
    word_count: int
    available: bool = True


class TranscriptResponse(BaseModel):
    """Full transcript response."""

    meeting_date: str
    segments: list[TranscriptSegment]
    total_segments: int
    powell_word_count: int


class TranscriptsListResponse(BaseModel):
    """List of available transcripts."""

    transcripts: list[TranscriptSummary]
    total_transcripts: int


# --- Backtest ---


class FOMCBacktestTrade(BaseModel):
    """A single trade from FOMC backtest."""

    meeting_date: str
    contract: str
    prediction_date: str
    days_before_meeting: int
    side: str
    position_size: float
    entry_price: float
    predicted_probability: float
    edge: float
    actual_outcome: int
    pnl: float
    roi: float


class FOMCHorizonMetrics(BaseModel):
    """Performance metrics for a specific time horizon."""

    horizon_days: int
    total_predictions: int
    correct_predictions: int
    accuracy: float
    total_trades: int
    winning_trades: int
    win_rate: float
    total_pnl: float
    avg_pnl_per_trade: float
    roi: float
    sharpe_ratio: float
    brier_score: float


class FOMCBacktestResponse(BaseModel):
    """Complete FOMC backtest results."""

    horizon_metrics: list[FOMCHorizonMetrics]
    overall_metrics: dict
    trades: list[FOMCBacktestTrade]
    metadata: dict


# --- Contracts ---


class FOMCContractSchema(BaseModel):
    """Schema for a single FOMC Kalshi contract."""

    market_ticker: str
    word: str
    threshold: int = 1
    status: str
    last_price: Optional[float] = None
    yes_bid: Optional[float] = None
    yes_ask: Optional[float] = None
    expiration_time: Optional[str] = None
    result: Optional[str] = None


class FOMCContractsResponse(BaseModel):
    """Response containing FOMC contracts from Kalshi."""

    contracts: list[FOMCContractSchema]
    active_count: int
    settled_count: int
    total_count: int


# --- Edges ---


class FOMCEdgeOpportunity(BaseModel):
    """A trading opportunity with positive edge."""

    contract: str
    predicted_probability: float
    market_price: float
    edge: float
    signal: str
    confidence_lower: float
    confidence_upper: float
    market_ticker: Optional[str] = None


class FOMCEdgesResponse(BaseModel):
    """Response containing edge opportunities."""

    opportunities: list[FOMCEdgeOpportunity]
    total_contracts_scanned: int
    generated_at: str
