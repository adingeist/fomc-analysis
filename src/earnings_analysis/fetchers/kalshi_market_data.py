"""
Fetch historical market data from Kalshi earnings mention contracts.

This module provides:
1. Historical price fetching for active and finalized contracts
2. Outcome extraction from finalized contracts
3. Local caching to avoid repeated API calls
4. DataFrames compatible with backtester's market_prices parameter
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, asdict, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from fomc_analysis.kalshi_client_factory import KalshiClientProtocol, KalshiSdkAdapter


@dataclass
class ContractMarketData:
    """Market data for a single Kalshi contract."""
    ticker: str  # Kalshi market ticker (e.g., "KXEARNINGSMENTIONMETA-26JUN30-VR")
    series_ticker: str  # Series ticker (e.g., "KXEARNINGSMENTIONMETA")
    company_ticker: str  # Stock ticker (e.g., "META")
    word: str  # Tracked word (e.g., "VR / Virtual Reality")

    # Contract details
    call_date: str  # Earnings call date (extracted from ticker)
    expiration_time: str  # Contract expiration
    status: str  # "active", "finalized", "settled", etc.

    # Price data
    last_price: float  # Last traded price (0-1)
    yes_bid: Optional[float] = None  # Current YES bid
    yes_ask: Optional[float] = None  # Current YES ask
    volume: Optional[int] = None  # Trading volume

    # Outcome (for finalized contracts)
    outcome: Optional[int] = None  # 1 = YES, 0 = NO (for finalized)

    # Metadata
    fetched_at: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class KalshiMarketSnapshot:
    """Snapshot of market data for all contracts of a ticker."""
    company_ticker: str
    contracts: List[ContractMarketData]
    fetched_at: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> dict:
        return {
            "company_ticker": self.company_ticker,
            "contracts": [asdict(c) for c in self.contracts],
            "fetched_at": self.fetched_at,
        }


class KalshiMarketDataFetcher:
    """
    Fetch and cache Kalshi earnings contract market data.

    This fetcher:
    1. Retrieves current market prices from active contracts
    2. Extracts outcomes from finalized contracts
    3. Caches data locally to minimize API calls
    4. Produces DataFrames compatible with the backtester

    Parameters
    ----------
    kalshi_client : KalshiClientProtocol
        Kalshi API client
    cache_dir : Path
        Directory for caching fetched data
    """

    def __init__(
        self,
        kalshi_client: KalshiClientProtocol,
        cache_dir: Optional[Path] = None,
    ):
        self.client = kalshi_client
        self.cache_dir = Path(cache_dir) if cache_dir else Path("data/kalshi_cache")
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def fetch_market_data(
        self,
        ticker: str,
        use_cache: bool = True,
        cache_max_age_hours: int = 1,
    ) -> KalshiMarketSnapshot:
        """
        Fetch market data for all contracts of a ticker.

        Parameters
        ----------
        ticker : str
            Company stock ticker (e.g., "META", "TSLA")
        use_cache : bool
            Whether to use cached data if available
        cache_max_age_hours : int
            Maximum age of cache in hours

        Returns
        -------
        KalshiMarketSnapshot
            Snapshot of all contract market data
        """
        ticker = ticker.upper()
        cache_file = self.cache_dir / f"{ticker}_market_data.json"

        # Check cache
        if use_cache and cache_file.exists():
            cached = self._load_cache(cache_file)
            if cached and self._is_cache_valid(cached, cache_max_age_hours):
                print(f"Using cached market data for {ticker}")
                return self._parse_cached_snapshot(cached)

        # Fetch from API
        print(f"Fetching market data for {ticker} from Kalshi API...")
        contracts = self._fetch_all_contracts(ticker)

        snapshot = KalshiMarketSnapshot(
            company_ticker=ticker,
            contracts=contracts,
        )

        # Save to cache
        self._save_cache(cache_file, snapshot.to_dict())

        return snapshot

    def _fetch_all_contracts(self, ticker: str) -> List[ContractMarketData]:
        """Fetch all contracts (active and finalized) for a ticker."""
        series_ticker = f"KXEARNINGSMENTION{ticker}"

        all_contracts = []

        # Fetch all markets for this series
        try:
            if isinstance(self.client, KalshiSdkAdapter):
                # Use async method if available
                import asyncio
                markets = asyncio.get_event_loop().run_until_complete(
                    self.client.get_markets_async(series_ticker=series_ticker)
                )
            else:
                markets = self.client.get_markets(series_ticker=series_ticker)
        except Exception as e:
            print(f"Error fetching markets for {series_ticker}: {e}")
            return []

        if not markets:
            print(f"No markets found for {series_ticker}")
            return []

        print(f"Found {len(markets)} contracts for {ticker}")

        for market in markets:
            contract_data = self._parse_market_to_contract(market, ticker, series_ticker)
            if contract_data:
                all_contracts.append(contract_data)

        return all_contracts

    def _parse_market_to_contract(
        self,
        market: Dict[str, Any],
        company_ticker: str,
        series_ticker: str,
    ) -> Optional[ContractMarketData]:
        """Parse a Kalshi market response into ContractMarketData."""
        market_ticker = market.get("ticker", "")
        status = market.get("status", "").lower()

        # Extract word from custom_strike or yes_sub_title
        word = self._extract_word(market)
        if not word:
            return None

        # Extract call date from ticker
        # Format: KXEARNINGSMENTIONMETA-26JUN30-VR
        call_date = self._extract_call_date(market_ticker)

        # Get prices (convert from cents to decimal)
        last_price = market.get("last_price", 50) / 100.0
        yes_bid = market.get("yes_bid")
        yes_ask = market.get("yes_ask")
        volume = market.get("volume")

        if yes_bid is not None:
            yes_bid = yes_bid / 100.0
        if yes_ask is not None:
            yes_ask = yes_ask / 100.0

        # Determine outcome for finalized contracts
        outcome = None
        if status in ("finalized", "settled", "resolved"):
            # For finalized contracts, last_price indicates outcome
            # Close to 100 = YES won, close to 0 = NO won
            # Using 50 as threshold (but in practice, finalized shows 99 or 1)
            outcome = 1 if last_price > 0.5 else 0

        return ContractMarketData(
            ticker=market_ticker,
            series_ticker=series_ticker,
            company_ticker=company_ticker,
            word=word,
            call_date=call_date,
            expiration_time=market.get("expiration_time", ""),
            status=status,
            last_price=last_price,
            yes_bid=yes_bid,
            yes_ask=yes_ask,
            volume=volume,
            outcome=outcome,
        )

    def _extract_word(self, market: Dict[str, Any]) -> Optional[str]:
        """Extract the tracked word from a market."""
        # Try custom_strike.Word first
        custom_strike = market.get("custom_strike", {})
        if isinstance(custom_strike, dict):
            word = custom_strike.get("Word")
            if word:
                return word.lower()

        # Try yes_sub_title
        yes_sub_title = market.get("yes_sub_title", "")
        if yes_sub_title:
            return yes_sub_title.lower()

        # Try to parse from title
        title = market.get("title", "")
        match = re.search(r"(?:say|mention)\s+['\"](.+?)['\"]", title, re.IGNORECASE)
        if match:
            return match.group(1).lower()

        return None

    def _extract_call_date(self, market_ticker: str) -> str:
        """Extract earnings call date from market ticker."""
        # Format: KXEARNINGSMENTIONMETA-26JUN30-VR
        # Date part: 26JUN30 = June 30, 2026
        match = re.search(r"-(\d{2})([A-Z]{3})(\d{2})-", market_ticker)
        if match:
            year = int(match.group(1)) + 2000  # Assumes 20xx
            month_str = match.group(2)
            day = int(match.group(3))

            # Convert month abbreviation
            months = {
                "JAN": 1, "FEB": 2, "MAR": 3, "APR": 4,
                "MAY": 5, "JUN": 6, "JUL": 7, "AUG": 8,
                "SEP": 9, "OCT": 10, "NOV": 11, "DEC": 12,
            }
            month = months.get(month_str.upper(), 1)

            return f"{year}-{month:02d}-{day:02d}"

        return ""

    def get_market_prices_df(
        self,
        ticker: str,
        use_cache: bool = True,
    ) -> pd.DataFrame:
        """
        Get market prices as DataFrame for backtester.

        Parameters
        ----------
        ticker : str
            Company stock ticker
        use_cache : bool
            Whether to use cached data

        Returns
        -------
        pd.DataFrame
            Index = call_date, Columns = word names, Values = prices (0-1)
        """
        snapshot = self.fetch_market_data(ticker, use_cache=use_cache)

        if not snapshot.contracts:
            return pd.DataFrame()

        # Group by call_date and word
        data = {}
        for contract in snapshot.contracts:
            call_date = contract.call_date
            word = contract.word
            price = contract.last_price

            if not call_date:
                continue

            if call_date not in data:
                data[call_date] = {}
            data[call_date][word] = price

        if not data:
            return pd.DataFrame()

        df = pd.DataFrame.from_dict(data, orient="index")
        df.index = pd.to_datetime(df.index)
        df = df.sort_index()

        return df

    def get_outcomes_df(
        self,
        ticker: str,
        use_cache: bool = True,
    ) -> pd.DataFrame:
        """
        Get contract outcomes as DataFrame (ground truth).

        Only includes finalized contracts with known outcomes.

        Parameters
        ----------
        ticker : str
            Company stock ticker
        use_cache : bool
            Whether to use cached data

        Returns
        -------
        pd.DataFrame
            Index = call_date, Columns = word names, Values = 0/1 outcomes
        """
        snapshot = self.fetch_market_data(ticker, use_cache=use_cache)

        if not snapshot.contracts:
            return pd.DataFrame()

        # Filter to finalized contracts with outcomes
        finalized = [c for c in snapshot.contracts
                     if c.status in ("finalized", "settled", "resolved")
                     and c.outcome is not None]

        if not finalized:
            print(f"No finalized contracts found for {ticker}")
            return pd.DataFrame()

        # Group by call_date and word
        data = {}
        for contract in finalized:
            call_date = contract.call_date
            word = contract.word

            if not call_date:
                continue

            if call_date not in data:
                data[call_date] = {}
            data[call_date][word] = contract.outcome

        if not data:
            return pd.DataFrame()

        df = pd.DataFrame.from_dict(data, orient="index")
        df.index = pd.to_datetime(df.index)
        df = df.sort_index()

        return df

    def get_active_contracts(
        self,
        ticker: str,
        use_cache: bool = True,
    ) -> List[ContractMarketData]:
        """
        Get currently active contracts for trading.

        Parameters
        ----------
        ticker : str
            Company stock ticker
        use_cache : bool
            Whether to use cached data

        Returns
        -------
        List[ContractMarketData]
            Active contracts
        """
        snapshot = self.fetch_market_data(ticker, use_cache=use_cache)
        return [c for c in snapshot.contracts if c.status == "active"]

    def _load_cache(self, cache_file: Path) -> Optional[dict]:
        """Load cached data from file."""
        try:
            with open(cache_file, "r") as f:
                return json.load(f)
        except Exception:
            return None

    def _save_cache(self, cache_file: Path, data: dict) -> None:
        """Save data to cache file."""
        with open(cache_file, "w") as f:
            json.dump(data, f, indent=2, default=str)
        print(f"Cached market data to {cache_file}")

    def _is_cache_valid(self, cached: dict, max_age_hours: int) -> bool:
        """Check if cached data is still valid."""
        fetched_at = cached.get("fetched_at")
        if not fetched_at:
            return False

        try:
            cached_time = datetime.fromisoformat(fetched_at)
            age_hours = (datetime.now() - cached_time).total_seconds() / 3600
            return age_hours < max_age_hours
        except Exception:
            return False

    def _parse_cached_snapshot(self, cached: dict) -> KalshiMarketSnapshot:
        """Parse cached data back into KalshiMarketSnapshot."""
        contracts = []
        for c in cached.get("contracts", []):
            contracts.append(ContractMarketData(**c))

        return KalshiMarketSnapshot(
            company_ticker=cached.get("company_ticker", ""),
            contracts=contracts,
            fetched_at=cached.get("fetched_at", ""),
        )


def fetch_kalshi_market_data(
    ticker: str,
    cache_dir: Optional[Path] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Convenience function to fetch Kalshi market data.

    Parameters
    ----------
    ticker : str
        Company stock ticker (e.g., "META", "TSLA")
    cache_dir : Optional[Path]
        Directory for caching data

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame]
        (market_prices_df, outcomes_df)
        - market_prices_df: Index=call_date, Columns=words, Values=prices
        - outcomes_df: Index=call_date, Columns=words, Values=0/1 outcomes
    """
    from fomc_analysis.kalshi_client_factory import get_kalshi_client

    client = get_kalshi_client()
    fetcher = KalshiMarketDataFetcher(client, cache_dir=cache_dir)

    market_prices = fetcher.get_market_prices_df(ticker)
    outcomes = fetcher.get_outcomes_df(ticker)

    return market_prices, outcomes


def fetch_multi_ticker_market_data(
    tickers: List[str],
    cache_dir: Optional[Path] = None,
) -> Dict[str, Tuple[pd.DataFrame, pd.DataFrame]]:
    """
    Fetch market data for multiple tickers.

    Parameters
    ----------
    tickers : List[str]
        List of company stock tickers
    cache_dir : Optional[Path]
        Directory for caching data

    Returns
    -------
    Dict[str, Tuple[pd.DataFrame, pd.DataFrame]]
        Mapping of ticker -> (market_prices_df, outcomes_df)
    """
    from fomc_analysis.kalshi_client_factory import get_kalshi_client

    client = get_kalshi_client()
    fetcher = KalshiMarketDataFetcher(client, cache_dir=cache_dir)

    results = {}
    for ticker in tickers:
        print(f"\nFetching data for {ticker}...")
        market_prices = fetcher.get_market_prices_df(ticker)
        outcomes = fetcher.get_outcomes_df(ticker)
        results[ticker] = (market_prices, outcomes)

    return results
