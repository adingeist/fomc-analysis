"""
Trade data fetcher for Kalshi earnings mention contracts.

Fetches trade-level data from the Kalshi REST API for microstructure
analysis. Supports cursor-based pagination, checkpoint/resume, and
batch processing across multiple tickers.

API endpoint: GET /trade-api/v2/markets/trades
Docs: https://trading-api.readme.io/reference/gettrades
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import requests

from .trade_storage import ParquetStorage


# Kalshi public API base URL
_KALSHI_API_BASE = "https://api.elections.kalshi.com/trade-api/v2"

# Standard earnings tickers
EARNINGS_TICKERS = [
    "META", "TSLA", "NVDA", "AAPL", "MSFT", "AMZN", "GOOGL", "NFLX",
]

# Rate limiting
_REQUEST_DELAY = 0.5  # Seconds between requests
_MAX_RETRIES = 3
_RETRY_BACKOFF = 2.0


@dataclass
class FetchProgress:
    """Track backfill progress for checkpoint/resume."""
    series_ticker: str
    cursor: Optional[str] = None
    total_fetched: int = 0
    complete: bool = False

    def to_dict(self) -> Dict:
        return {
            "series_ticker": self.series_ticker,
            "cursor": self.cursor,
            "total_fetched": self.total_fetched,
            "complete": self.complete,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "FetchProgress":
        return cls(**data)


class EarningsTradesFetcher:
    """
    Fetch trade-level data for earnings mention contracts from Kalshi.

    Supports:
    - Cursor-based pagination through all trades
    - Checkpoint/resume for interrupted backfills
    - Rate limiting to respect API constraints
    - Batch processing across multiple tickers

    Parameters
    ----------
    storage : ParquetStorage
        Where to store fetched trades.
    checkpoint_dir : Path
        Directory for checkpoint files (resume support).
    request_delay : float
        Seconds between API requests.
    batch_size : int
        Records per API request (max 1000).
    """

    def __init__(
        self,
        storage: Optional[ParquetStorage] = None,
        checkpoint_dir: Optional[Path] = None,
        request_delay: float = _REQUEST_DELAY,
        batch_size: int = 1000,
    ):
        self.storage = storage or ParquetStorage()
        self.checkpoint_dir = Path(checkpoint_dir or "data/microstructure/checkpoints")
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.request_delay = request_delay
        self.batch_size = min(batch_size, 1000)  # API max is 1000
        self.session = requests.Session()

    def _checkpoint_path(self, series_ticker: str) -> Path:
        return self.checkpoint_dir / f"{series_ticker}_progress.json"

    def _load_checkpoint(self, series_ticker: str) -> Optional[FetchProgress]:
        path = self._checkpoint_path(series_ticker)
        if path.exists():
            data = json.loads(path.read_text())
            return FetchProgress.from_dict(data)
        return None

    def _save_checkpoint(self, progress: FetchProgress):
        path = self._checkpoint_path(progress.series_ticker)
        path.write_text(json.dumps(progress.to_dict(), indent=2))

    def _fetch_trades_page(
        self,
        ticker: str,
        cursor: Optional[str] = None,
    ) -> Dict:
        """
        Fetch a single page of trades from the Kalshi API.

        Parameters
        ----------
        ticker : str
            Market ticker to fetch trades for.
        cursor : str, optional
            Pagination cursor from previous response.

        Returns
        -------
        dict
            API response with 'trades' list and optional 'cursor'.
        """
        params = {
            "ticker": ticker,
            "limit": self.batch_size,
        }
        if cursor:
            params["cursor"] = cursor

        for attempt in range(_MAX_RETRIES):
            try:
                resp = self.session.get(
                    f"{_KALSHI_API_BASE}/markets/trades",
                    params=params,
                    timeout=30,
                )
                resp.raise_for_status()
                return resp.json()
            except (requests.RequestException, json.JSONDecodeError) as e:
                if attempt < _MAX_RETRIES - 1:
                    wait = _RETRY_BACKOFF ** (attempt + 1)
                    print(f"  Retry {attempt + 1}/{_MAX_RETRIES} after {wait}s: {e}")
                    time.sleep(wait)
                else:
                    raise

    def _fetch_markets_page(
        self,
        series_ticker: str,
        cursor: Optional[str] = None,
        status: Optional[str] = None,
    ) -> Dict:
        """Fetch a page of markets from the Kalshi API."""
        params = {
            "series_ticker": series_ticker,
            "limit": 200,
        }
        if cursor:
            params["cursor"] = cursor
        if status:
            params["status"] = status

        for attempt in range(_MAX_RETRIES):
            try:
                resp = self.session.get(
                    f"{_KALSHI_API_BASE}/markets",
                    params=params,
                    timeout=30,
                )
                resp.raise_for_status()
                return resp.json()
            except (requests.RequestException, json.JSONDecodeError) as e:
                if attempt < _MAX_RETRIES - 1:
                    wait = _RETRY_BACKOFF ** (attempt + 1)
                    time.sleep(wait)
                else:
                    raise

    def fetch_all_markets(
        self,
        series_ticker: str,
        status: Optional[str] = None,
    ) -> List[Dict]:
        """
        Fetch all markets for a series, paginating through all results.

        Parameters
        ----------
        series_ticker : str
            e.g., "KXEARNINGSMENTIONMETA"
        status : str, optional
            Filter by status ("active", "finalized", etc.)

        Returns
        -------
        list of dict
            All matching markets.
        """
        all_markets = []
        cursor = None

        while True:
            data = self._fetch_markets_page(series_ticker, cursor, status)
            markets = data.get("markets", [])
            all_markets.extend(markets)

            cursor = data.get("cursor")
            if not cursor or not markets:
                break

            time.sleep(self.request_delay)

        return all_markets

    def fetch_ticker_trades(
        self,
        market_ticker: str,
        resume: bool = True,
    ) -> int:
        """
        Fetch all trades for a specific market ticker.

        Parameters
        ----------
        market_ticker : str
            Specific market ticker (e.g., "KXEARNINGSMENTIONMETA-26JUN30-VR").
        resume : bool
            Whether to resume from checkpoint.

        Returns
        -------
        int
            Number of new trades fetched.
        """
        series_ticker = market_ticker.rsplit("-", 2)[0] if "-" in market_ticker else market_ticker

        progress = None
        if resume:
            progress = self._load_checkpoint(market_ticker)
            if progress and progress.complete:
                print(f"  {market_ticker}: already complete ({progress.total_fetched} trades)")
                return 0

        if progress is None:
            progress = FetchProgress(series_ticker=market_ticker)

        total_new = 0
        cursor = progress.cursor

        while True:
            data = self._fetch_trades_page(market_ticker, cursor)
            trades = data.get("trades", [])

            if not trades:
                progress.complete = True
                self._save_checkpoint(progress)
                break

            # Save batch
            saved = self.storage.save_trades(series_ticker, trades)
            total_new += saved
            progress.total_fetched += len(trades)

            cursor = data.get("cursor")
            progress.cursor = cursor
            self._save_checkpoint(progress)

            if not cursor:
                progress.complete = True
                self._save_checkpoint(progress)
                break

            time.sleep(self.request_delay)

        return total_new

    def backfill_earnings_trades(
        self,
        companies: Optional[List[str]] = None,
        min_volume: int = 10,
        resume: bool = True,
    ) -> Dict[str, int]:
        """
        Backfill trade data for all earnings mention contracts.

        For each company:
        1. Fetch all markets (active + finalized)
        2. Filter to markets with sufficient volume
        3. Fetch all trades for each market

        Parameters
        ----------
        companies : list of str, optional
            Company tickers. Defaults to EARNINGS_TICKERS.
        min_volume : int
            Minimum volume to fetch trades (default: 10).
        resume : bool
            Resume from checkpoints.

        Returns
        -------
        dict
            Mapping of series_ticker -> total new trades.
        """
        companies = companies or EARNINGS_TICKERS
        results = {}

        for company in companies:
            series_ticker = f"KXEARNINGSMENTION{company}"
            print(f"\n{'='*60}")
            print(f"Backfilling: {series_ticker}")
            print(f"{'='*60}")

            # Fetch all markets for this series
            markets = self.fetch_all_markets(series_ticker)
            print(f"  Found {len(markets)} markets")

            # Save market metadata
            if markets:
                market_records = []
                for m in markets:
                    market_records.append({
                        "ticker": m.get("ticker", ""),
                        "series_ticker": series_ticker,
                        "title": m.get("title", ""),
                        "status": m.get("status", ""),
                        "yes_bid": m.get("yes_bid", 0),
                        "yes_ask": m.get("yes_ask", 0),
                        "last_price": m.get("last_price", 0),
                        "volume": m.get("volume", 0),
                        "open_interest": m.get("open_interest", 0),
                        "result": m.get("result", ""),
                        "expiration_time": m.get("expiration_time", ""),
                        "custom_strike_word": (
                            m.get("custom_strike", {}).get("Word", "")
                            if m.get("custom_strike") else ""
                        ),
                    })
                self.storage.save_markets(
                    market_records,
                    filename=f"{series_ticker}_markets.parquet",
                )

            # Fetch trades for each market with enough volume
            total_new = 0
            eligible = [m for m in markets if m.get("volume", 0) >= min_volume]
            print(f"  {len(eligible)} markets with volume >= {min_volume}")

            for i, market in enumerate(eligible):
                market_ticker = market.get("ticker", "")
                volume = market.get("volume", 0)
                print(f"  [{i+1}/{len(eligible)}] {market_ticker} (vol={volume})")

                try:
                    new = self.fetch_ticker_trades(market_ticker, resume=resume)
                    total_new += new
                    if new > 0:
                        print(f"    Fetched {new} new trades")
                except Exception as e:
                    print(f"    ERROR: {e}")

            results[series_ticker] = total_new
            print(f"  Total new trades for {series_ticker}: {total_new}")

        return results

    def backfill_market_snapshots(
        self,
        companies: Optional[List[str]] = None,
    ) -> int:
        """
        Take a snapshot of current market prices for all earnings contracts.

        This is a lightweight operation that captures bid/ask/last/volume
        for efficiency monitoring over time.

        Parameters
        ----------
        companies : list of str, optional
            Company tickers.

        Returns
        -------
        int
            Number of market records saved.
        """
        companies = companies or EARNINGS_TICKERS
        all_records = []

        for company in companies:
            series_ticker = f"KXEARNINGSMENTION{company}"
            markets = self.fetch_all_markets(series_ticker, status="active")

            for m in markets:
                all_records.append({
                    "ticker": m.get("ticker", ""),
                    "series_ticker": series_ticker,
                    "company": company,
                    "word": (
                        m.get("custom_strike", {}).get("Word", "")
                        if m.get("custom_strike") else ""
                    ),
                    "yes_bid": m.get("yes_bid", 0),
                    "yes_ask": m.get("yes_ask", 0),
                    "last_price": m.get("last_price", 0),
                    "volume": m.get("volume", 0),
                    "open_interest": m.get("open_interest", 0),
                })

            time.sleep(self.request_delay)

        saved = self.storage.save_snapshot(all_records, "market_prices")
        print(f"Saved {saved} market price records")
        return saved
