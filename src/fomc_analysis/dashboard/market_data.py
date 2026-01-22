"""
Live market data service for fetching real-time prices from Kalshi.

This module provides functionality to fetch current market prices for FOMC
prediction contracts using the Kalshi SDK.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Optional

from kalshi_python_async import KalshiClient

from ..kalshi_sdk import create_kalshi_sdk_client


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


class MarketDataService:
    """Service for fetching live market data from Kalshi."""

    def __init__(self, client: Optional[KalshiClient] = None):
        """Initialize the market data service.

        Args:
            client: Optional KalshiClient instance. If not provided,
                   a new client will be created using credentials from settings.
        """
        self.client = client or create_kalshi_sdk_client()

    async def get_market_price(self, ticker: str) -> MarketPrice:
        """Fetch live price data for a single market.

        Args:
            ticker: Market ticker (e.g., "KXFEDMENTION-26JAN-INFLATION40")

        Returns:
            MarketPrice object with current pricing data
        """
        market_data = await self.client.get_market(ticker=ticker)
        return MarketPrice.from_api_response(market_data.market.model_dump())

    async def get_markets_prices(
        self, event_ticker: Optional[str] = None, tickers: Optional[list[str]] = None
    ) -> list[MarketPrice]:
        """Fetch live prices for multiple markets.

        Args:
            event_ticker: Filter by event ticker (e.g., "KXFEDMENTION-26JAN")
            tickers: List of specific tickers to fetch

        Returns:
            List of MarketPrice objects
        """
        params = {"status": "open", "limit": 1000}

        if event_ticker:
            params["event_ticker"] = event_ticker
        if tickers:
            params["tickers"] = ",".join(tickers)

        markets_response = await self.client.get_markets(**params)
        return [
            MarketPrice.from_api_response(market.model_dump())
            for market in markets_response.markets
        ]

    def get_market_price_sync(self, ticker: str) -> MarketPrice:
        """Synchronous wrapper for getting market price.

        Args:
            ticker: Market ticker

        Returns:
            MarketPrice object
        """
        return asyncio.run(self.get_market_price(ticker))

    def get_markets_prices_sync(
        self, event_ticker: Optional[str] = None, tickers: Optional[list[str]] = None
    ) -> list[MarketPrice]:
        """Synchronous wrapper for getting multiple market prices.

        Args:
            event_ticker: Filter by event ticker
            tickers: List of specific tickers to fetch

        Returns:
            List of MarketPrice objects
        """
        return asyncio.run(self.get_markets_prices(event_ticker, tickers))


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
        service = MarketDataService()
        prices = service.get_markets_prices_sync(tickers=unique_tickers)
        return {price.ticker: price for price in prices}
    except Exception as e:
        print(f"Error fetching live prices: {e}")
        return {}
