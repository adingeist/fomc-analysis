"""Pydantic request/response schemas for the Earnings API."""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel


# --- Predictions ---


class WordPrediction(BaseModel):
    word: str
    probability: float
    raw_probability: float
    lower_bound: float
    upper_bound: float
    uncertainty: float
    n_samples: int
    market_price: Optional[float] = None
    edge: Optional[float] = None
    adjusted_edge: Optional[float] = None
    trade_signal: str = "HOLD"
    kelly_fraction: float = 0.0
    confidence: Optional[float] = None


class PredictionResponse(BaseModel):
    ticker: str
    predictions: list[WordPrediction]
    model_type: str
    generated_at: str


# --- Edges ---


class EdgeOpportunity(BaseModel):
    ticker: str
    word: str
    predicted_probability: float
    market_price: float
    edge: float
    adjusted_edge: float
    signal: str
    kelly_fraction: float
    confidence: Optional[float] = None
    yes_bid: Optional[float] = None
    yes_ask: Optional[float] = None
    market_ticker: str = ""


class EdgesResponse(BaseModel):
    ticker: Optional[str] = None
    opportunities: list[EdgeOpportunity]
    total_contracts_scanned: int
    generated_at: str


# --- Backtests ---


class BacktestTradeSchema(BaseModel):
    ticker: str
    call_date: str
    contract: str
    side: str
    position_size: float
    entry_price: float
    predicted_probability: float
    edge: float
    actual_outcome: int
    pnl: float
    roi: float


class BacktestMetricsSchema(BaseModel):
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


class BacktestResponse(BaseModel):
    ticker: str
    metrics: BacktestMetricsSchema
    trades: list[BacktestTradeSchema]
    metadata: dict


# --- Contracts ---


class ContractSchema(BaseModel):
    market_ticker: str
    word: str
    status: str
    last_price: Optional[float] = None
    yes_bid: Optional[float] = None
    yes_ask: Optional[float] = None
    expiration_time: Optional[str] = None


class ContractsResponse(BaseModel):
    ticker: str
    contracts: list[ContractSchema]
    active_count: int
    total_count: int


# --- Word Frequencies ---


class EarningsWordFrequencyRecord(BaseModel):
    call_date: str
    word: str
    count: int
    mentioned: bool


class EarningsWordFrequencySeries(BaseModel):
    word: str
    frequencies: list[EarningsWordFrequencyRecord]
    total_mentions: int
    mention_rate: float


class EarningsWordFrequenciesResponse(BaseModel):
    ticker: str
    words: list[EarningsWordFrequencySeries]
    call_dates: list[str]
    total_calls: int


# --- Transcripts ---


class EarningsTranscriptSegment(BaseModel):
    segment_idx: int
    speaker: str
    role: str
    text: str


class EarningsTranscriptSummary(BaseModel):
    ticker: str
    call_date: str
    total_segments: int
    executive_segments: int
    word_count: int
    available: bool = True


class EarningsTranscriptResponse(BaseModel):
    ticker: str
    call_date: str
    segments: list[EarningsTranscriptSegment]
    total_segments: int
    executive_word_count: int


class EarningsTranscriptsListResponse(BaseModel):
    ticker: str
    transcripts: list[EarningsTranscriptSummary]
    total_transcripts: int


# --- Health ---


class ModelStatus(BaseModel):
    ticker: str
    words: list[str]
    trained: bool
    n_words: int


class HealthResponse(BaseModel):
    status: str
    models_loaded: int
    tickers: list[ModelStatus]
    kalshi_connected: bool
