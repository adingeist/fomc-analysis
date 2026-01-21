"""
kalshi_api
==========

This module provides a lightweight wrapper around the Kalshi API for
downloading historical price data and, optionally, executing trades.
At the time of writing the API requires authentication via an API
key and secret.  See https://docs.kalshi.com/ for details.

Important: this module is a minimal example.  It does not handle
rate limiting, retries, or the full breadth of the API.  You should
review and adapt it for your own use, especially for production
trading.  Additionally, this module does not execute any trades
itself; it only downloads data.  Execution functions are left as
stubs.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Optional

import pandas as pd
import requests

from .config import settings

KALSHI_BASE_URL = "https://trading-api.kalshi.com/trade-api/v2"


@dataclass
class KalshiClient:
    """Simple client for the Kalshi API.

    Parameters
    ----------
    api_key: Optional[str]
        Your Kalshi API key.  Obtain from your Kalshi account.
        If not provided, will attempt to load from environment variables
        (KALSHI_API_KEY) or .env file via Pydantic settings.
    api_secret: Optional[str]
        Your Kalshi API secret.  Obtain from your Kalshi account.
        If not provided, will attempt to load from environment variables
        (KALSHI_API_SECRET) or .env file via Pydantic settings.
    base_url: Optional[str]
        Base URL for the API.  If not provided, uses KALSHI_BASE_URL from
        settings or defaults to the production API URL.
        You can override this for testing.
    session: Optional[requests.Session]
        Optional HTTP session to reuse TCP connections.

    Notes
    -----
    The API currently uses HTTP Basic authentication.  This client
    attaches the credentials to each request.  If Kalshi changes
    their authentication scheme (e.g. bearer tokens), you will need
    to update this code.

    Credentials can be provided in three ways (in order of precedence):
    1. Directly as constructor arguments
    2. Environment variables (KALSHI_API_KEY, KALSHI_API_SECRET)
    3. .env file in the project root

    Raises
    ------
    ValueError
        If neither api_key/api_secret are provided directly nor found
        in environment variables or .env file.
    """

    api_key: Optional[str] = None
    api_secret: Optional[str] = None
    base_url: Optional[str] = None
    session: Optional[requests.Session] = None

    def __post_init__(self) -> None:
        # Load credentials from settings if not provided directly
        if self.api_key is None:
            self.api_key = settings.kalshi_api_key
        if self.api_secret is None:
            self.api_secret = settings.kalshi_api_secret
        if self.base_url is None:
            self.base_url = settings.kalshi_base_url or KALSHI_BASE_URL

        # Validate that we have credentials
        if not self.api_key or not self.api_secret:
            raise ValueError(
                "Kalshi API credentials are required. "
                "Provide them as constructor arguments (api_key, api_secret), "
                "or set environment variables KALSHI_API_KEY and KALSHI_API_SECRET, "
                "or add them to a .env file in the project root."
            )

        if self.session is None:
            self.session = requests.Session()
        self.session.auth = (self.api_key, self.api_secret)

    def _request(self, method: str, path: str, **kwargs: Any) -> Dict[str, Any]:
        """Internal helper to send an authenticated HTTP request."""
        url = f"{self.base_url}{path}"
        resp = self.session.request(method, url, **kwargs)
        resp.raise_for_status()
        return resp.json()

    def get_market_history(
        self,
        ticker: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> pd.DataFrame:
        """Fetch historical price data for a given market ticker.

        Parameters
        ----------
        ticker: str
            The Kalshi market ticker (e.g. "KXFEDMENTION-26JAN").
        start_date: Optional[str], format YYYY-MM-DD
            If provided, fetch data starting from this date.
        end_date: Optional[str], format YYYY-MM-DD
            If provided, fetch data up to this date (inclusive).

        Returns
        -------
        pandas.DataFrame
            DataFrame with columns ``date`` and ``price`` (YES price in
            cents).  The date column is of dtype datetime64[ns].

        Notes
        -----
        The API returns a list of price updates.  This function
        aggregates them by date (taking the last update of each day)
        because transcripts are usually analysed at daily resolution.
        You can modify this behaviour to preserve higher resolution.
        """
        params = {}
        if start_date:
            params["start_date"] = start_date
        if end_date:
            params["end_date"] = end_date
        data = self._request("GET", f"/markets/{ticker}/history", params=params)
        # The API returns a dictionary with a key 'history' containing a list of records.
        history = data.get("history", [])
        records = []
        for rec in history:
            # rec may have 'date' and 'lastPrice'; adjust if API fields differ
            date_str = rec.get("date") or rec.get("timestamp")
            price = rec.get("last_price") or rec.get("price") or rec.get("close")
            if date_str is None or price is None:
                continue
            records.append((datetime.fromisoformat(date_str), float(price)))
        if not records:
            return pd.DataFrame(columns=["date", "price"])
        df = pd.DataFrame(records, columns=["date", "price"])
        # Group by date (day) and take last price of the day
        df["day"] = df["date"].dt.date
        daily = df.groupby("day").agg({"price": "last"})
        daily.index = pd.to_datetime(daily.index)
        daily = daily.rename(columns={"price": ticker})
        return daily

    def get_series(self, series_ticker: str) -> Dict[str, Any]:
        """Fetch information about a series.

        Parameters
        ----------
        series_ticker: str
            The series ticker (e.g., "KXFEDMENTION").

        Returns
        -------
        Dict[str, Any]
            Series information including title, frequency, contract terms, etc.
        """
        data = self._request("GET", f"/series/{series_ticker}")
        return data.get("series", {})

    def get_event(
        self, event_ticker: str, with_nested_markets: bool = True
    ) -> Dict[str, Any]:
        """Fetch information about an event.

        Parameters
        ----------
        event_ticker: str
            The event ticker (e.g., "kxfedmention-26jan").
        with_nested_markets: bool, default=True
            If True, include nested market data in the response.

        Returns
        -------
        Dict[str, Any]
            Event information including title, markets, strike date, etc.
        """
        params = {"with_nested_markets": str(with_nested_markets).lower()}
        data = self._request("GET", f"/events/{event_ticker}", params=params)
        return data

    def get_markets(
        self,
        series_ticker: Optional[str] = None,
        event_ticker: Optional[str] = None,
        status: Optional[str] = None,
        limit: int = 200,
    ) -> list[Dict[str, Any]]:
        """Fetch markets with optional filtering.

        Parameters
        ----------
        series_ticker: Optional[str]
            Filter by series ticker.
        event_ticker: Optional[str]
            Filter by event ticker.
        status: Optional[str]
            Filter by status (e.g., "open", "closed").
        limit: int, default=200
            Maximum number of markets to return.

        Returns
        -------
        list[Dict[str, Any]]
            List of market objects.
        """
        params = {"limit": limit}
        if series_ticker:
            params["series_ticker"] = series_ticker
        if event_ticker:
            params["event_ticker"] = event_ticker
        if status:
            params["status"] = status

        data = self._request("GET", "/markets", params=params)
        return data.get("markets", [])

    # ------------------------------------------------------------------
    # Placeholder methods for trade execution
    # ------------------------------------------------------------------
    def place_order(
        self, ticker: str, side: str, size: float, price: float
    ) -> Dict[str, Any]:
        """Stub for placing an order on Kalshi.

        This method is not implemented.  Trading on Kalshi requires
        regulatory compliance and careful error handling.  Implement
        this method at your own risk.
        """
        raise NotImplementedError(
            "Order execution is not implemented in this example.  "
            "Please implement your own trading logic if desired."
        )
