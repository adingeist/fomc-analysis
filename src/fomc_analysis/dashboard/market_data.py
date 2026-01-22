"""
Live market data service for fetching real-time prices from Kalshi.

This module provides functionality to fetch current market prices for FOMC
prediction contracts using the Kalshi SDK.
"""

from __future__ import annotations

import asyncio
import inspect
from dataclasses import dataclass
from typing import Optional

from ..kalshi_client_factory import KalshiSdkAdapter

_TICKER_BATCH_SIZE = 50
_DEFAULT_MARKET_LIMIT = 1000


@dataclass
class MarketPrice:
    """Live market price data for a contract."""

    ticker: str
    yes_bid: float | None = None
    yes_ask: float | None = None
    no_bid: float | None = None
    no_ask: float | None = None
    last_price: float | None = None
    mid_price: float | None = None
    volume: int | None = None
    volume_24h: int | None = None
    open_interest: int | None = None
    status: str | None = None

    @classmethod
    def from_api_response(cls, market_data: dict) -> "MarketPrice":
        """Create MarketPrice from Kalshi API response."""
        ticker = market_data.get("ticker", "")

        # Extract price data (convert from cents to 0-1 range)
        yes_bid = cls._parse_price(market_data.get("yes_bid"))
        yes_ask = cls._parse_price(market_data.get("yes_ask"))
        no_bid = cls._parse_price(market_data.get("no_bid"))
        no_ask = cls._parse_price(market_data.get("no_ask"))
        last_price = cls._parse_price(market_data.get("last_price"))

        # Calculate mid price (average of yes_bid and yes_ask)
        mid_price = None
        if yes_bid is not None and yes_ask is not None:
            mid_price = (yes_bid + yes_ask) / 2
        elif last_price is not None:
            mid_price = last_price

        return cls(
            ticker=ticker,
            yes_bid=yes_bid,
            yes_ask=yes_ask,
            no_bid=no_bid,
            no_ask=no_ask,
            last_price=last_price,
            mid_price=mid_price,
            volume=market_data.get("volume"),
            volume_24h=market_data.get("volume_24h"),
            open_interest=market_data.get("open_interest"),
            status=market_data.get("status"),
        )

    @staticmethod
    def _parse_price(price_value: Optional[int | float]) -> float | None:
        """Convert price from cents to 0-1 range."""
        if price_value is None:
            return None
        # Kalshi prices are in cents (0-100), convert to 0-1
        return float(price_value) / 100.0


def _chunked(items: list[str], chunk_size: int):
    """Yield successive chunks from a list."""
    for i in range(0, len(items), chunk_size):
        yield items[i : i + chunk_size]


class MarketDataService:
    """Service for fetching live market data from Kalshi."""

    def __init__(self, client: Optional[object] = None):
        """Initialize the market data service.

        Args:
            client: Optional Kalshi SDK adapter instance. If not provided,
                a new adapter will be created using credentials from settings.
        """
        self.client = client or KalshiSdkAdapter()
        self._owns_client = client is None

    def __enter__(self) -> "MarketDataService":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    async def __aenter__(self) -> "MarketDataService":
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self.aclose()

    def close(self) -> None:
        """Close the underlying client if this service created it."""
        if not self._owns_client:
            return
        close_method = getattr(self.client, "close", None)
        if not callable(close_method):
            return
        result = close_method()
        if inspect.isawaitable(result):
            asyncio.run(result)

    async def aclose(self) -> None:
        """Async variant of close for use with async context managers."""
        if not self._owns_client:
            return
        close_method = getattr(self.client, "close", None)
        if not callable(close_method):
            return
        result = close_method()
        if inspect.isawaitable(result):
            await result

    async def get_market_price(self, ticker: str) -> MarketPrice:
        """Fetch live price data for a single market."""
        return self.get_market_price_sync(ticker)

    async def get_markets_prices(
        self, event_ticker: Optional[str] = None, tickers: Optional[list[str]] = None
    ) -> list[MarketPrice]:
        """Fetch live prices for multiple markets."""
        return self.get_markets_prices_sync(event_ticker, tickers)

    def _call_get_markets(self, **kwargs):
        try:
            return self.client.get_markets(**kwargs)
        except TypeError:
            if "tickers" in kwargs:
                kwargs.pop("tickers")
                return self.client.get_markets(**kwargs)
            raise

    def _normalize_markets(
        self, response, expected_tickers: Optional[list[str]] = None
    ) -> list[dict]:
        if isinstance(response, dict):
            response = response.get("markets", [])
        if not isinstance(response, list):
            return []
        desired = set(expected_tickers) if expected_tickers else None
        normalized: list[dict] = []
        for market in response:
            if hasattr(market, "model_dump"):
                market_data = market.model_dump()
            elif isinstance(market, dict):
                market_data = market
            else:
                continue
            if desired and market_data.get("ticker") not in desired:
                continue
            normalized.append(market_data)
        return normalized

    def _fetch_markets_data(
        self, event_ticker: Optional[str], tickers: Optional[list[str]]
    ) -> list[dict]:
        limit = _DEFAULT_MARKET_LIMIT
        if tickers:
            limit = max(len(tickers), 1)
        params = {"status": "open", "limit": limit}
        if event_ticker:
            params["event_ticker"] = event_ticker
        if tickers:
            params["tickers"] = tickers
        response = self._call_get_markets(**params)
        return self._normalize_markets(response, tickers)

    def get_market_price_sync(self, ticker: str) -> MarketPrice:
        """Fetch a single market price synchronously."""
        prices = self.get_markets_prices_sync(tickers=[ticker])
        return prices[0] if prices else MarketPrice(ticker=ticker)

    def get_markets_prices_sync(
        self, event_ticker: Optional[str] = None, tickers: Optional[list[str]] = None
    ) -> list[MarketPrice]:
        """Fetch multiple market prices synchronously."""
        if tickers:
            price_data: list[MarketPrice] = []
            for chunk in _chunked(tickers, _TICKER_BATCH_SIZE):
                markets = self._fetch_markets_data(event_ticker, chunk)
                price_data.extend(
                    MarketPrice.from_api_response(market) for market in markets
                )
            return price_data

        markets = self._fetch_markets_data(event_ticker, None)
        return [MarketPrice.from_api_response(market) for market in markets]


def fetch_live_prices_for_predictions(predictions_df) -> dict[str, MarketPrice]:
    """Fetch live prices for all tickers in a predictions DataFrame.

    Args:
        predictions_df: DataFrame with 'ticker' column

    Returns:
        Dictionary mapping ticker to MarketPrice
    """
    if predictions_df.empty or "ticker" not in predictions_df.columns:
        return {}

    unique_tickers = predictions_df["ticker"].dropna().unique().tolist()
    if not unique_tickers:
        return {}

    try:
        with MarketDataService() as service:
            prices = service.get_markets_prices_sync(tickers=unique_tickers)
            return {price.ticker: price for price in prices}
    except Exception as e:
        print(f"Error fetching live prices: {e}")
        return {}
