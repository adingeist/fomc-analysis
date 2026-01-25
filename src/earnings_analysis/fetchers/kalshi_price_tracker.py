"""
Historical price tracking for Kalshi earnings mention contracts.

This module provides:
1. Daily price snapshot collection for active contracts
2. Historical price storage in CSV format
3. Price evolution analysis (divergence from model predictions)
4. Integration with backtester for time-aware price lookups

Usage:
    from earnings_analysis.fetchers.kalshi_price_tracker import KalshiPriceTracker

    tracker = KalshiPriceTracker()
    tracker.record_snapshot("META")  # Take daily snapshot

    # Get historical prices for backtesting
    prices_df = tracker.get_historical_prices("META", "ai")
"""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict, field
from datetime import datetime, date
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import numpy as np

from fomc_analysis.kalshi_client_factory import KalshiClientProtocol, get_kalshi_client


@dataclass
class PriceSnapshot:
    """A single price snapshot for a contract."""
    # Identification
    snapshot_date: str  # Date snapshot was taken (YYYY-MM-DD)
    ticker: str  # Kalshi market ticker
    company_ticker: str  # Stock ticker (e.g., "META")
    word: str  # Tracked word (normalized lowercase)

    # Timing
    call_date: str  # Earnings call date
    expiration_time: str  # Contract expiration
    days_to_expiry: int  # Days until expiration

    # Price data
    last_price: float  # Last traded price (0-1)
    yes_bid: Optional[float] = None
    yes_ask: Optional[float] = None
    mid_price: Optional[float] = None  # (bid + ask) / 2
    spread: Optional[float] = None  # ask - bid
    volume: Optional[int] = None

    # Status
    status: str = "active"

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class PriceHistory:
    """Historical price data for a contract."""
    company_ticker: str
    word: str
    call_date: str
    snapshots: List[PriceSnapshot] = field(default_factory=list)

    def to_dataframe(self) -> pd.DataFrame:
        """Convert snapshots to DataFrame."""
        if not self.snapshots:
            return pd.DataFrame()

        records = [s.to_dict() for s in self.snapshots]
        df = pd.DataFrame(records)
        df["snapshot_date"] = pd.to_datetime(df["snapshot_date"])
        df = df.sort_values("snapshot_date")
        return df


class KalshiPriceTracker:
    """
    Track and store historical Kalshi contract prices.

    This tracker:
    1. Takes daily snapshots of contract prices
    2. Stores data in CSV format for easy analysis
    3. Calculates price evolution metrics
    4. Provides historical prices for backtesting

    Parameters
    ----------
    kalshi_client : KalshiClientProtocol, optional
        Kalshi API client (creates one if not provided)
    data_dir : Path, optional
        Directory for storing price history (default: data/kalshi_prices)
    """

    def __init__(
        self,
        kalshi_client: Optional[KalshiClientProtocol] = None,
        data_dir: Optional[Path] = None,
    ):
        self.client = kalshi_client or get_kalshi_client()
        self.data_dir = Path(data_dir) if data_dir else Path("data/kalshi_prices")
        self.data_dir.mkdir(parents=True, exist_ok=True)

    def record_snapshot(
        self,
        ticker: str,
        snapshot_date: Optional[str] = None,
    ) -> List[PriceSnapshot]:
        """
        Record current prices for all contracts of a ticker.

        Parameters
        ----------
        ticker : str
            Company stock ticker (e.g., "META", "TSLA")
        snapshot_date : str, optional
            Date to record as (default: today's date)

        Returns
        -------
        List[PriceSnapshot]
            List of recorded price snapshots
        """
        ticker = ticker.upper()
        snapshot_date = snapshot_date or date.today().isoformat()

        print(f"Recording price snapshot for {ticker} on {snapshot_date}...")

        # Fetch current market data
        series_ticker = f"KXEARNINGSMENTION{ticker}"

        try:
            markets = self.client.get_markets(series_ticker=series_ticker)
        except Exception as e:
            print(f"Error fetching markets for {series_ticker}: {e}")
            return []

        if not markets:
            print(f"No markets found for {series_ticker}")
            return []

        snapshots = []

        for market in markets:
            snapshot = self._create_snapshot(market, ticker, snapshot_date)
            if snapshot:
                snapshots.append(snapshot)

        # Save to CSV
        if snapshots:
            self._append_to_csv(ticker, snapshots)
            print(f"Recorded {len(snapshots)} price snapshots for {ticker}")

        return snapshots

    def _create_snapshot(
        self,
        market: Dict[str, Any],
        company_ticker: str,
        snapshot_date: str,
    ) -> Optional[PriceSnapshot]:
        """Create a PriceSnapshot from market data."""
        market_ticker = market.get("ticker", "")
        status = market.get("status", "").lower()

        # Only track active contracts
        if status != "active":
            return None

        # Extract word
        word = self._extract_word(market)
        if not word:
            return None

        # Extract call date from ticker
        call_date = self._extract_call_date(market_ticker)
        if not call_date:
            return None

        # Get expiration time
        expiration_time = market.get("expiration_time", "")

        # Calculate days to expiry
        days_to_expiry = self._calculate_days_to_expiry(snapshot_date, expiration_time)

        # Get prices (convert from cents to decimal)
        last_price = market.get("last_price", 50) / 100.0
        yes_bid = market.get("yes_bid")
        yes_ask = market.get("yes_ask")
        volume = market.get("volume")

        if yes_bid is not None:
            yes_bid = yes_bid / 100.0
        if yes_ask is not None:
            yes_ask = yes_ask / 100.0

        # Calculate mid price and spread
        mid_price = None
        spread = None
        if yes_bid is not None and yes_ask is not None:
            mid_price = (yes_bid + yes_ask) / 2
            spread = yes_ask - yes_bid

        return PriceSnapshot(
            snapshot_date=snapshot_date,
            ticker=market_ticker,
            company_ticker=company_ticker,
            word=word,
            call_date=call_date,
            expiration_time=expiration_time,
            days_to_expiry=days_to_expiry,
            last_price=last_price,
            yes_bid=yes_bid,
            yes_ask=yes_ask,
            mid_price=mid_price,
            spread=spread,
            volume=volume,
            status=status,
        )

    def _extract_word(self, market: Dict[str, Any]) -> Optional[str]:
        """Extract the tracked word from a market."""
        custom_strike = market.get("custom_strike", {})
        if isinstance(custom_strike, dict):
            word = custom_strike.get("Word")
            if word:
                return word.lower()

        yes_sub_title = market.get("yes_sub_title", "")
        if yes_sub_title:
            return yes_sub_title.lower()

        return None

    def _extract_call_date(self, market_ticker: str) -> str:
        """Extract earnings call date from market ticker."""
        import re
        # Format: KXEARNINGSMENTIONMETA-26JUN30-VR
        match = re.search(r"-(\d{2})([A-Z]{3})(\d{2})-", market_ticker)
        if match:
            year = int(match.group(1)) + 2000
            month_str = match.group(2)
            day = int(match.group(3))

            months = {
                "JAN": 1, "FEB": 2, "MAR": 3, "APR": 4,
                "MAY": 5, "JUN": 6, "JUL": 7, "AUG": 8,
                "SEP": 9, "OCT": 10, "NOV": 11, "DEC": 12,
            }
            month = months.get(month_str.upper(), 1)

            return f"{year}-{month:02d}-{day:02d}"

        return ""

    def _calculate_days_to_expiry(
        self,
        snapshot_date: str,
        expiration_time: str,
    ) -> int:
        """Calculate days until contract expiration."""
        try:
            snap_date = datetime.fromisoformat(snapshot_date).date()

            # Parse expiration time (ISO format)
            if expiration_time:
                exp_date = datetime.fromisoformat(
                    expiration_time.replace("Z", "+00:00")
                ).date()
                return (exp_date - snap_date).days
        except (ValueError, TypeError):
            pass

        return 0

    def _append_to_csv(self, ticker: str, snapshots: List[PriceSnapshot]) -> None:
        """Append snapshots to CSV file."""
        csv_file = self.data_dir / f"{ticker}_price_history.csv"

        new_df = pd.DataFrame([s.to_dict() for s in snapshots])

        if csv_file.exists():
            # Load existing and append
            existing_df = pd.read_csv(csv_file)

            # Remove duplicates (same snapshot_date + ticker + word)
            existing_df["_key"] = (
                existing_df["snapshot_date"] + "_" +
                existing_df["ticker"] + "_" +
                existing_df["word"]
            )
            new_df["_key"] = (
                new_df["snapshot_date"] + "_" +
                new_df["ticker"] + "_" +
                new_df["word"]
            )

            # Keep existing records that aren't being updated
            existing_df = existing_df[~existing_df["_key"].isin(new_df["_key"])]

            # Combine and sort
            combined_df = pd.concat([existing_df, new_df], ignore_index=True)
            combined_df = combined_df.drop(columns=["_key"])
            combined_df = combined_df.sort_values(["word", "snapshot_date"])
            combined_df.to_csv(csv_file, index=False)
        else:
            new_df.to_csv(csv_file, index=False)

        print(f"Saved price history to {csv_file}")

    def get_price_history(
        self,
        ticker: str,
        word: Optional[str] = None,
        call_date: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Get historical price data from CSV.

        Parameters
        ----------
        ticker : str
            Company stock ticker
        word : str, optional
            Filter to specific word
        call_date : str, optional
            Filter to specific earnings call date

        Returns
        -------
        pd.DataFrame
            Price history with columns:
            snapshot_date, word, last_price, yes_bid, yes_ask,
            mid_price, spread, volume, days_to_expiry
        """
        ticker = ticker.upper()
        csv_file = self.data_dir / f"{ticker}_price_history.csv"

        if not csv_file.exists():
            print(f"No price history found for {ticker}")
            return pd.DataFrame()

        df = pd.read_csv(csv_file)
        df["snapshot_date"] = pd.to_datetime(df["snapshot_date"])

        if word:
            df = df[df["word"] == word.lower()]

        if call_date:
            df = df[df["call_date"] == call_date]

        return df.sort_values(["word", "snapshot_date"])

    def get_historical_prices_for_backtest(
        self,
        ticker: str,
        as_of_date: str,
    ) -> Dict[str, float]:
        """
        Get market prices as of a specific date for backtesting.

        This returns the most recent price snapshot on or before as_of_date
        for each word, preventing look-ahead bias.

        Parameters
        ----------
        ticker : str
            Company stock ticker
        as_of_date : str
            Date to get prices for (YYYY-MM-DD)

        Returns
        -------
        Dict[str, float]
            Mapping of word -> price (0-1)
        """
        df = self.get_price_history(ticker)

        if df.empty:
            return {}

        # Filter to snapshots on or before as_of_date
        as_of_dt = pd.to_datetime(as_of_date)
        df = df[df["snapshot_date"] <= as_of_dt]

        if df.empty:
            return {}

        # Get most recent snapshot for each word
        latest_idx = df.groupby("word")["snapshot_date"].idxmax()
        latest = df.loc[latest_idx]

        return dict(zip(latest["word"], latest["last_price"]))

    def get_price_at_date(
        self,
        ticker: str,
        word: str,
        target_date: str,
    ) -> Optional[float]:
        """
        Get price for a specific word on a specific date.

        Parameters
        ----------
        ticker : str
            Company stock ticker
        word : str
            Word to look up
        target_date : str
            Date to get price for

        Returns
        -------
        float or None
            Price if available, None otherwise
        """
        df = self.get_price_history(ticker, word=word)

        if df.empty:
            return None

        target_dt = pd.to_datetime(target_date)

        # Try exact match first
        exact = df[df["snapshot_date"] == target_dt]
        if not exact.empty:
            return float(exact.iloc[0]["last_price"])

        # Fall back to most recent before target
        before = df[df["snapshot_date"] < target_dt]
        if not before.empty:
            return float(before.iloc[-1]["last_price"])

        return None


class PriceEvolutionAnalyzer:
    """
    Analyze price evolution and model divergence.

    This analyzer:
    1. Tracks how market prices evolve over time
    2. Compares model predictions to market consensus
    3. Identifies optimal trading windows
    4. Calculates divergence metrics
    """

    def __init__(
        self,
        price_tracker: KalshiPriceTracker,
    ):
        self.tracker = price_tracker

    def analyze_price_evolution(
        self,
        ticker: str,
        word: str,
        call_date: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Analyze how prices evolve as expiration approaches.

        Parameters
        ----------
        ticker : str
            Company stock ticker
        word : str
            Word to analyze
        call_date : str, optional
            Specific earnings call to analyze

        Returns
        -------
        pd.DataFrame
            Price evolution metrics:
            - snapshot_date
            - days_to_expiry
            - last_price
            - price_change_1d (daily change)
            - price_change_cumulative (from first snapshot)
            - volatility_5d (5-day rolling std)
        """
        df = self.tracker.get_price_history(ticker, word=word, call_date=call_date)

        if df.empty:
            return pd.DataFrame()

        # Sort by date
        df = df.sort_values("snapshot_date").reset_index(drop=True)

        # Calculate metrics
        df["price_change_1d"] = df["last_price"].diff()
        df["price_change_cumulative"] = df["last_price"] - df["last_price"].iloc[0]
        df["volatility_5d"] = df["last_price"].rolling(window=5, min_periods=2).std()

        return df

    def calculate_divergence(
        self,
        ticker: str,
        model_predictions: Dict[str, float],
        as_of_date: str,
    ) -> pd.DataFrame:
        """
        Calculate divergence between model and market prices.

        Parameters
        ----------
        ticker : str
            Company stock ticker
        model_predictions : Dict[str, float]
            Model's predicted probabilities (word -> probability)
        as_of_date : str
            Date for market price comparison

        Returns
        -------
        pd.DataFrame
            Divergence analysis:
            - word
            - model_prob
            - market_price
            - divergence (model - market)
            - abs_divergence
            - suggested_side ("YES" if divergence > 0, "NO" otherwise)
        """
        market_prices = self.tracker.get_historical_prices_for_backtest(
            ticker, as_of_date
        )

        records = []
        for word, model_prob in model_predictions.items():
            market_price = market_prices.get(word.lower())

            if market_price is not None:
                divergence = model_prob - market_price
                records.append({
                    "word": word,
                    "model_prob": model_prob,
                    "market_price": market_price,
                    "divergence": divergence,
                    "abs_divergence": abs(divergence),
                    "suggested_side": "YES" if divergence > 0 else "NO",
                })

        if not records:
            return pd.DataFrame()

        df = pd.DataFrame(records)
        df = df.sort_values("abs_divergence", ascending=False)
        return df

    def find_optimal_trading_window(
        self,
        ticker: str,
        word: str,
    ) -> Dict[str, Any]:
        """
        Analyze historical data to find optimal trading window.

        Parameters
        ----------
        ticker : str
            Company stock ticker
        word : str
            Word to analyze

        Returns
        -------
        dict
            Analysis results:
            - avg_volatility_by_days_to_expiry
            - avg_spread_by_days_to_expiry
            - recommended_window (days before expiry)
            - analysis_notes
        """
        df = self.tracker.get_price_history(ticker, word=word)

        if df.empty or len(df) < 5:
            return {
                "error": "Insufficient price history",
                "min_snapshots_required": 5,
                "snapshots_found": len(df),
            }

        # Group by days_to_expiry buckets
        df["days_bucket"] = pd.cut(
            df["days_to_expiry"],
            bins=[0, 3, 7, 14, 30, float("inf")],
            labels=["0-3d", "4-7d", "8-14d", "15-30d", "30d+"]
        )

        # Calculate metrics per bucket
        bucket_stats = df.groupby("days_bucket", observed=True).agg({
            "spread": ["mean", "std"],
            "last_price": ["std", "count"],
        }).round(4)

        # Find bucket with best liquidity (lowest spread, reasonable volume)
        avg_spreads = df.groupby("days_bucket", observed=True)["spread"].mean()

        # Recommend window with lowest spread (best liquidity)
        if not avg_spreads.empty:
            best_bucket = avg_spreads.idxmin()
            recommended_window = {
                "0-3d": "1-3 days before expiry",
                "4-7d": "4-7 days before expiry",
                "8-14d": "8-14 days before expiry",
                "15-30d": "15-30 days before expiry",
                "30d+": "More than 30 days before expiry",
            }.get(best_bucket, "Unknown")
        else:
            recommended_window = "Insufficient data"

        return {
            "ticker": ticker,
            "word": word,
            "total_snapshots": len(df),
            "bucket_statistics": bucket_stats.to_dict() if not bucket_stats.empty else {},
            "avg_spread_by_bucket": avg_spreads.to_dict() if not avg_spreads.empty else {},
            "recommended_window": recommended_window,
            "analysis_notes": [
                "Lower spread = better liquidity for trading",
                "Consider price stability when choosing entry point",
                "Avoid trading too close to expiry (high volatility)",
            ],
        }

    def track_divergence_over_time(
        self,
        ticker: str,
        word: str,
        model_predictions: pd.Series,
    ) -> pd.DataFrame:
        """
        Track how divergence evolves as expiration approaches.

        Parameters
        ----------
        ticker : str
            Company stock ticker
        word : str
            Word to track
        model_predictions : pd.Series
            Model predictions indexed by date

        Returns
        -------
        pd.DataFrame
            Divergence time series with columns:
            - snapshot_date
            - market_price
            - model_prediction
            - divergence
            - days_to_expiry
        """
        df = self.tracker.get_price_history(ticker, word=word)

        if df.empty:
            return pd.DataFrame()

        # Merge with model predictions
        df = df.set_index("snapshot_date")

        # Align model predictions
        model_df = model_predictions.to_frame("model_prediction")
        merged = df.join(model_df, how="left")

        # Forward fill model predictions (assume constant until updated)
        merged["model_prediction"] = merged["model_prediction"].ffill()

        # Calculate divergence
        merged["divergence"] = merged["model_prediction"] - merged["last_price"]

        return merged.reset_index()[[
            "snapshot_date",
            "last_price",
            "model_prediction",
            "divergence",
            "days_to_expiry",
        ]]


def record_daily_prices(
    tickers: List[str],
    data_dir: Optional[Path] = None,
    snapshot_date: Optional[str] = None,
) -> Dict[str, int]:
    """
    Convenience function to record daily prices for multiple tickers.

    Parameters
    ----------
    tickers : List[str]
        List of company stock tickers
    data_dir : Path, optional
        Directory for storing price history
    snapshot_date : str, optional
        Date to record as (default: today)

    Returns
    -------
    Dict[str, int]
        Mapping of ticker -> number of snapshots recorded
    """
    tracker = KalshiPriceTracker(data_dir=data_dir)

    results = {}
    for ticker in tickers:
        print(f"\n--- Recording prices for {ticker} ---")
        snapshots = tracker.record_snapshot(ticker, snapshot_date=snapshot_date)
        results[ticker] = len(snapshots)

    return results
