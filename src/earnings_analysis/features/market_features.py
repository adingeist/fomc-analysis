"""
Market-based features for earnings prediction.

Integrates:
1. Historical Kalshi market prices (market consensus)
2. Stock price momentum and volatility
3. Earnings surprise history
4. Prior call mention patterns
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    import yfinance as yf
    HAS_YFINANCE = True
except ImportError:
    HAS_YFINANCE = False


@dataclass
class MarketFeatures:
    """Market-derived features for a single prediction."""
    # Kalshi market features
    market_price: float  # Current market consensus (0-1)
    market_price_7d_ago: Optional[float] = None  # Price 7 days ago
    market_momentum: Optional[float] = None  # Price change (momentum)
    market_volume: Optional[float] = None  # Trading volume (if available)

    # Stock features
    stock_return_30d: Optional[float] = None  # 30-day return before earnings
    stock_volatility_30d: Optional[float] = None  # 30-day volatility
    stock_return_5d: Optional[float] = None  # 5-day return (short-term)

    # Earnings history features
    eps_surprise_last: Optional[float] = None  # Last earnings surprise %
    eps_surprise_avg_4q: Optional[float] = None  # Avg surprise last 4 quarters
    eps_beat_streak: Optional[int] = None  # Consecutive beats

    # Prior call features
    mentioned_last_call: Optional[int] = None  # Was word mentioned last quarter
    mention_rate_4q: Optional[float] = None  # Mention rate last 4 quarters
    mention_trend: Optional[float] = None  # Trend in mention frequency


class MarketFeatureExtractor:
    """
    Extract market-based features for earnings predictions.

    These features capture:
    - What the market already knows (Kalshi prices)
    - Stock-level context (momentum, volatility)
    - Earnings quality signals (surprise history)
    - Word-level patterns (prior mentions)
    """

    def __init__(
        self,
        cache_dir: Optional[Path] = None,
        use_cached: bool = True,
    ):
        self.cache_dir = Path(cache_dir) if cache_dir else Path("data/market_cache")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.use_cached = use_cached

        # Cache for expensive API calls
        self._stock_cache: Dict[str, pd.DataFrame] = {}
        self._earnings_cache: Dict[str, pd.DataFrame] = {}

    def get_stock_data(
        self,
        ticker: str,
        start_date: str,
        end_date: str,
    ) -> Optional[pd.DataFrame]:
        """Fetch stock price data using yfinance."""
        if not HAS_YFINANCE:
            return None

        cache_key = f"{ticker}_{start_date}_{end_date}"

        if cache_key in self._stock_cache:
            return self._stock_cache[cache_key]

        # Check file cache
        cache_file = self.cache_dir / f"stock_{cache_key}.csv"
        if self.use_cached and cache_file.exists():
            df = pd.read_csv(cache_file, index_col=0, parse_dates=True)
            self._stock_cache[cache_key] = df
            return df

        try:
            stock = yf.Ticker(ticker)
            df = stock.history(start=start_date, end=end_date)

            if not df.empty:
                df.to_csv(cache_file)
                self._stock_cache[cache_key] = df

            return df if not df.empty else None

        except Exception as e:
            print(f"Error fetching stock data for {ticker}: {e}")
            return None

    def get_earnings_history(self, ticker: str) -> Optional[pd.DataFrame]:
        """Fetch earnings surprise history using yfinance."""
        if not HAS_YFINANCE:
            return None

        if ticker in self._earnings_cache:
            return self._earnings_cache[ticker]

        cache_file = self.cache_dir / f"earnings_{ticker}.csv"
        if self.use_cached and cache_file.exists():
            df = pd.read_csv(cache_file, index_col=0, parse_dates=True)
            self._earnings_cache[ticker] = df
            return df

        try:
            stock = yf.Ticker(ticker)
            earnings = stock.earnings_history

            if earnings is not None and not earnings.empty:
                earnings.to_csv(cache_file)
                self._earnings_cache[ticker] = earnings
                return earnings

            return None

        except Exception as e:
            print(f"Error fetching earnings for {ticker}: {e}")
            return None

    def calculate_stock_features(
        self,
        ticker: str,
        call_date: str,
        lookback_days: int = 60,
    ) -> Dict[str, Optional[float]]:
        """Calculate stock-based features before an earnings call."""
        features = {
            "stock_return_30d": None,
            "stock_volatility_30d": None,
            "stock_return_5d": None,
        }

        call_dt = pd.to_datetime(call_date)
        start_date = (call_dt - timedelta(days=lookback_days)).strftime("%Y-%m-%d")
        end_date = call_dt.strftime("%Y-%m-%d")

        df = self.get_stock_data(ticker, start_date, end_date)

        if df is None or len(df) < 10:
            return features

        # Calculate returns
        df["return"] = df["Close"].pct_change()

        # 30-day return (or available days)
        if len(df) >= 22:  # ~1 month of trading days
            features["stock_return_30d"] = (
                df["Close"].iloc[-1] / df["Close"].iloc[-22] - 1
            )

        # 5-day return
        if len(df) >= 5:
            features["stock_return_5d"] = (
                df["Close"].iloc[-1] / df["Close"].iloc[-5] - 1
            )

        # 30-day volatility (annualized)
        if len(df) >= 22:
            features["stock_volatility_30d"] = df["return"].iloc[-22:].std() * np.sqrt(252)

        return features

    def calculate_earnings_features(
        self,
        ticker: str,
        call_date: str,
    ) -> Dict[str, Optional[float]]:
        """Calculate earnings surprise features."""
        features = {
            "eps_surprise_last": None,
            "eps_surprise_avg_4q": None,
            "eps_beat_streak": None,
        }

        earnings = self.get_earnings_history(ticker)

        if earnings is None or len(earnings) < 1:
            return features

        # Filter to earnings before this call
        call_dt = pd.to_datetime(call_date)

        if "Earnings Date" in earnings.columns:
            earnings["date"] = pd.to_datetime(earnings["Earnings Date"])
        else:
            earnings["date"] = earnings.index

        prior_earnings = earnings[earnings["date"] < call_dt].copy()

        if len(prior_earnings) < 1:
            return features

        # Sort by date descending (most recent first)
        prior_earnings = prior_earnings.sort_values("date", ascending=False)

        # Calculate surprise percentage
        if "Surprise(%)" in prior_earnings.columns:
            surprises = prior_earnings["Surprise(%)"].dropna()
        elif "EPS Actual" in prior_earnings.columns and "EPS Estimate" in prior_earnings.columns:
            actual = prior_earnings["EPS Actual"]
            estimate = prior_earnings["EPS Estimate"]
            surprises = ((actual - estimate) / estimate.abs().clip(lower=0.01)) * 100
        else:
            return features

        if len(surprises) >= 1:
            features["eps_surprise_last"] = float(surprises.iloc[0])

        if len(surprises) >= 4:
            features["eps_surprise_avg_4q"] = float(surprises.iloc[:4].mean())

        # Beat streak
        beat_streak = 0
        for surprise in surprises:
            if surprise > 0:
                beat_streak += 1
            else:
                break
        features["eps_beat_streak"] = beat_streak

        return features

    def calculate_prior_mention_features(
        self,
        word: str,
        call_date: str,
        historical_outcomes: pd.DataFrame,
    ) -> Dict[str, Optional[float]]:
        """Calculate features based on prior mention patterns."""
        features = {
            "mentioned_last_call": None,
            "mention_rate_4q": None,
            "mention_trend": None,
        }

        if word not in historical_outcomes.columns:
            return features

        call_dt = pd.to_datetime(call_date)
        outcomes = historical_outcomes[word].copy()
        outcomes.index = pd.to_datetime(outcomes.index)

        # Filter to prior calls only
        prior_outcomes = outcomes[outcomes.index < call_dt].sort_index()

        if len(prior_outcomes) < 1:
            return features

        # Last call mention
        features["mentioned_last_call"] = int(prior_outcomes.iloc[-1])

        # 4-quarter mention rate
        if len(prior_outcomes) >= 4:
            features["mention_rate_4q"] = float(prior_outcomes.iloc[-4:].mean())
        elif len(prior_outcomes) >= 1:
            features["mention_rate_4q"] = float(prior_outcomes.mean())

        # Trend (recent vs older)
        if len(prior_outcomes) >= 4:
            recent = prior_outcomes.iloc[-2:].mean()
            older = prior_outcomes.iloc[-4:-2].mean()
            features["mention_trend"] = float(recent - older)

        return features

    def extract_features(
        self,
        ticker: str,
        word: str,
        call_date: str,
        market_price: float = 0.5,
        historical_outcomes: Optional[pd.DataFrame] = None,
        market_price_history: Optional[Dict[str, float]] = None,
    ) -> MarketFeatures:
        """
        Extract all market features for a single prediction.

        Parameters
        ----------
        ticker : str
            Stock ticker symbol
        word : str
            Word being predicted
        call_date : str
            Date of earnings call
        market_price : float
            Current Kalshi market price (0-1)
        historical_outcomes : pd.DataFrame
            Historical outcomes for prior mention features
        market_price_history : Dict[str, float]
            Historical market prices keyed by date

        Returns
        -------
        MarketFeatures
            Complete feature set
        """
        # Stock features
        stock_features = self.calculate_stock_features(ticker, call_date)

        # Earnings features
        earnings_features = self.calculate_earnings_features(ticker, call_date)

        # Prior mention features
        mention_features = {}
        if historical_outcomes is not None:
            mention_features = self.calculate_prior_mention_features(
                word, call_date, historical_outcomes
            )

        # Market momentum
        market_momentum = None
        market_price_7d_ago = None
        if market_price_history:
            call_dt = pd.to_datetime(call_date)
            week_ago = (call_dt - timedelta(days=7)).strftime("%Y-%m-%d")

            if week_ago in market_price_history:
                market_price_7d_ago = market_price_history[week_ago]
                market_momentum = market_price - market_price_7d_ago

        return MarketFeatures(
            market_price=market_price,
            market_price_7d_ago=market_price_7d_ago,
            market_momentum=market_momentum,
            market_volume=None,  # Would need Kalshi volume data
            stock_return_30d=stock_features.get("stock_return_30d"),
            stock_volatility_30d=stock_features.get("stock_volatility_30d"),
            stock_return_5d=stock_features.get("stock_return_5d"),
            eps_surprise_last=earnings_features.get("eps_surprise_last"),
            eps_surprise_avg_4q=earnings_features.get("eps_surprise_avg_4q"),
            eps_beat_streak=earnings_features.get("eps_beat_streak"),
            mentioned_last_call=mention_features.get("mentioned_last_call"),
            mention_rate_4q=mention_features.get("mention_rate_4q"),
            mention_trend=mention_features.get("mention_trend"),
        )

    def extract_features_dataframe(
        self,
        ticker: str,
        words: List[str],
        call_dates: List[str],
        historical_outcomes: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """
        Extract market features for multiple words and dates.

        Returns DataFrame with MultiIndex (call_date, word) and feature columns.
        """
        records = []

        for call_date in call_dates:
            for word in words:
                features = self.extract_features(
                    ticker=ticker,
                    word=word,
                    call_date=call_date,
                    market_price=0.5,  # Default; would be replaced with real prices
                    historical_outcomes=historical_outcomes,
                )

                record = {
                    "call_date": call_date,
                    "word": word,
                    "market_price": features.market_price,
                    "stock_return_30d": features.stock_return_30d,
                    "stock_volatility_30d": features.stock_volatility_30d,
                    "stock_return_5d": features.stock_return_5d,
                    "eps_surprise_last": features.eps_surprise_last,
                    "eps_surprise_avg_4q": features.eps_surprise_avg_4q,
                    "eps_beat_streak": features.eps_beat_streak,
                    "mentioned_last_call": features.mentioned_last_call,
                    "mention_rate_4q": features.mention_rate_4q,
                    "mention_trend": features.mention_trend,
                }
                records.append(record)

        df = pd.DataFrame(records)
        df = df.set_index(["call_date", "word"])

        return df


def create_combined_features(
    word_counts: pd.DataFrame,
    market_features: pd.DataFrame,
) -> pd.DataFrame:
    """
    Combine word count features with market features.

    Parameters
    ----------
    word_counts : pd.DataFrame
        Word counts per call (index = call_date, columns = words)
    market_features : pd.DataFrame
        Market features (MultiIndex = (call_date, word))

    Returns
    -------
    pd.DataFrame
        Combined features ready for model training
    """
    # Reshape word counts to long format
    word_counts_long = word_counts.reset_index().melt(
        id_vars=["index"],
        var_name="word",
        value_name="word_count"
    ).rename(columns={"index": "call_date"})

    word_counts_long = word_counts_long.set_index(["call_date", "word"])

    # Join with market features
    combined = word_counts_long.join(market_features, how="left")

    return combined
