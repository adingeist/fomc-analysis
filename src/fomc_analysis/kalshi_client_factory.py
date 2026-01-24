"""Helpers for constructing Kalshi API clients with multiple auth schemes."""

from __future__ import annotations

import asyncio
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Protocol

import pandas as pd

from .config import settings
from .kalshi_api import KalshiClient as LegacyKalshiClient
from .kalshi_sdk import create_kalshi_sdk_client

SDK_IMPORT_ERROR: Exception | None = None

try:
    from kalshi_python_async.api.market_api import MarketApi
    from kalshi_python_async.api.events_api import EventsApi
except Exception as exc:  # pragma: no cover - handled at runtime when dependency missing
    SDK_IMPORT_ERROR = exc
    MarketApi = None  # type: ignore
    EventsApi = None  # type: ignore

_DEFAULT_START_TS = int(datetime(2015, 1, 1, tzinfo=timezone.utc).timestamp())


def _current_timestamp() -> int:
    return int(datetime.now(tz=timezone.utc).timestamp())


def _date_str_to_timestamp(value: Optional[str], default: Optional[int]) -> int:
    if value:
        try:
            dt = datetime.fromisoformat(value)
        except ValueError:
            dt = None
        if dt is not None:
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return int(dt.timestamp())
    if default is not None:
        return default
    return _current_timestamp()


class KalshiClientProtocol(Protocol):
    """Minimal client interface consumed by the analyzer/CLI."""

    def get_markets(
        self,
        series_ticker: Optional[str] = None,
        event_ticker: Optional[str] = None,
        status: Optional[str] = None,
        limit: int = 200,
    ) -> List[Dict[str, Any]]:
        ...

    def get_event(
        self,
        event_ticker: str,
        with_nested_markets: bool = True,
    ) -> Dict[str, Any]:
        ...

    def get_series(self, series_ticker: str) -> Dict[str, Any]:
        ...

    def close(self) -> None:
        ...

    def get_market_history(
        self,
        ticker: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> pd.DataFrame:
        ...


class KalshiSdkAdapter:
    """Adapter exposing the same surface as LegacyKalshiClient via the SDK.

    Provides both synchronous and asynchronous methods for flexibility.
    Sync methods use an internal event loop for backward compatibility.
    Async methods can be used with asyncio.gather for parallel requests.
    """

    def __init__(self) -> None:
        if MarketApi is None or EventsApi is None:  # pragma: no cover - runtime guard
            message = (
                "kalshi_python_async is required for SDK authentication. "
                "Install dependencies with `uv sync` and retry."
            )
            if SDK_IMPORT_ERROR is not None:
                message += f" (root cause: {SDK_IMPORT_ERROR})"
            raise RuntimeError(message)

        self._loop = asyncio.new_event_loop()
        try:
            self._parent_loop = asyncio.get_event_loop()
        except RuntimeError:
            self._parent_loop = None

        asyncio.set_event_loop(self._loop)
        try:
            self._sdk_client = create_kalshi_sdk_client()
            self._market_api = MarketApi(self._sdk_client)
            self._events_api = EventsApi(self._sdk_client)
        finally:
            if self._parent_loop is not None:
                asyncio.set_event_loop(self._parent_loop)
            else:
                try:
                    asyncio.set_event_loop(None)
                except Exception:
                    pass

    def _run(self, coro):
        """Execute an SDK coroutine synchronously."""
        if self._loop.is_closed():
            raise RuntimeError("Kalshi SDK event loop is closed")
        return self._loop.run_until_complete(coro)

    def _await_json(self, coroutine):
        async def _inner():
            resp = await coroutine
            try:
                if hasattr(resp, "json"):
                    return await resp.json()
                data = await resp.read()
                import json as _json

                return _json.loads(data)
            finally:
                release = getattr(resp, "release", None)
                if callable(release):
                    release()

        return self._run(_inner())

    async def get_markets_async(
        self,
        series_ticker: Optional[str] = None,
        event_ticker: Optional[str] = None,
        status: Optional[str] = None,
        limit: int = 200,
        tickers: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """Async version of get_markets for use with asyncio.gather."""
        if tickers:
            tickers_param = ",".join(tickers)
        else:
            tickers_param = None

        resp = await self._market_api.get_markets_without_preload_content(
            series_ticker=series_ticker,
            event_ticker=event_ticker,
            status=status,
            limit=limit,
            tickers=tickers_param,
        )

        try:
            if hasattr(resp, "json"):
                payload = await resp.json()
            else:
                data = await resp.read()
                import json as _json
                payload = _json.loads(data)
        finally:
            release = getattr(resp, "release", None)
            if callable(release):
                release()

        if isinstance(payload, dict):
            markets = payload.get("markets", [])
            return markets or []
        return []

    def get_markets(
        self,
        series_ticker: Optional[str] = None,
        event_ticker: Optional[str] = None,
        status: Optional[str] = None,
        limit: int = 200,
        tickers: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """Synchronous version of get_markets (backward compatible)."""
        if tickers:
            tickers_param = ",".join(tickers)
        else:
            tickers_param = None
        payload = self._await_json(
            self._market_api.get_markets_without_preload_content(
                series_ticker=series_ticker,
                event_ticker=event_ticker,
                status=status,
                limit=limit,
                tickers=tickers_param,
            )
        )
        if isinstance(payload, dict):
            markets = payload.get("markets", [])
            return markets or []
        return []

    async def get_event_async(
        self,
        event_ticker: str,
        with_nested_markets: bool = True,
    ) -> Dict[str, Any]:
        """Async version of get_event for use with asyncio.gather."""
        resp = await self._events_api.get_event_without_preload_content(
            event_ticker=event_ticker,
            with_nested_markets=with_nested_markets,
        )

        try:
            if hasattr(resp, "json"):
                payload = await resp.json()
            else:
                data = await resp.read()
                import json as _json
                payload = _json.loads(data)
        finally:
            release = getattr(resp, "release", None)
            if callable(release):
                release()

        if isinstance(payload, dict):
            return payload
        return {"event": {}}

    def get_event(
        self,
        event_ticker: str,
        with_nested_markets: bool = True,
    ) -> Dict[str, Any]:
        """Synchronous version of get_event (backward compatible)."""
        payload = self._await_json(
            self._events_api.get_event_without_preload_content(
                event_ticker=event_ticker,
                with_nested_markets=with_nested_markets,
            )
        )
        if isinstance(payload, dict):
            return payload
        return {"event": {}}

    async def get_series_async(self, series_ticker: str) -> Dict[str, Any]:
        """Async version of get_series for use with asyncio.gather."""
        response = await self._market_api.get_series(series_ticker=series_ticker)
        if hasattr(response, "to_dict"):
            return response.to_dict()
        return {}

    def get_series(self, series_ticker: str) -> Dict[str, Any]:
        """Synchronous version of get_series (backward compatible)."""
        response = self._run(
            self._market_api.get_series(series_ticker=series_ticker)
        )
        if hasattr(response, "to_dict"):
            return response.to_dict()
        return {}

    def close(self) -> None:
        sdk_client = getattr(self, "_sdk_client", None)
        loop = getattr(self, "_loop", None)
        if sdk_client is None or loop is None:
            return
        try:
            loop.run_until_complete(sdk_client.close())
        finally:
            loop.close()

    def get_market_history(
        self,
        ticker: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> pd.DataFrame:
        series_ticker = ticker.split("-", 1)[0]
        start_ts = _date_str_to_timestamp(start_date, default=_DEFAULT_START_TS)
        end_ts = _date_str_to_timestamp(end_date, default=_current_timestamp())

        response = self._run(
            self._market_api.get_market_candlesticks(
                series_ticker=series_ticker,
                ticker=ticker,
                start_ts=start_ts,
                end_ts=end_ts,
                period_interval=1440,
            )
        )

        if hasattr(response, "to_dict"):
            payload = response.to_dict()
        else:
            payload = response

        candlesticks = payload.get("candlesticks", []) if isinstance(payload, dict) else []

        records: List[tuple] = []
        for candle in candlesticks:
            if isinstance(candle, dict):
                entry = candle
            elif hasattr(candle, "to_dict"):
                entry = candle.to_dict()
            else:
                continue
            ts = entry.get("end_period_ts") or entry.get("start_period_ts")
            price_info = entry.get("price") or {}
            close = price_info.get("close")
            if ts is None or close is None:
                continue
            dt = datetime.fromtimestamp(int(ts), tz=timezone.utc).date()
            records.append((pd.Timestamp(dt), float(close)))

        if not records:
            return pd.DataFrame(columns=[ticker])

        df = pd.DataFrame(records, columns=["date", ticker])
        df = df.drop_duplicates(subset="date").set_index("date").sort_index()
        return df


def has_legacy_credentials() -> bool:
    return bool(os.getenv("KALSHI_API_KEY") and os.getenv("KALSHI_API_SECRET"))


def has_sdk_credentials() -> bool:
    return bool(settings.kalshi_api_key_id and settings.kalshi_private_key_base64)


def get_kalshi_client() -> KalshiClientProtocol:
    """Return a Kalshi client using whichever credential set is configured."""

    if has_legacy_credentials():
        return LegacyKalshiClient()

    if has_sdk_credentials():
        try:
            return KalshiSdkAdapter()
        except RuntimeError as exc:
            hint = "Run `uv sync` (or `uv sync --extra dev`) to install dependencies."
            if SDK_IMPORT_ERROR is not None:
                hint += f" Root cause: {SDK_IMPORT_ERROR}"
            raise ValueError(hint) from exc

    raise ValueError(
        "Kalshi API credentials are required. Set either the legacy "
        "KALSHI_API_KEY/KALSHI_API_SECRET or the RSA pair "
        "KALSHI_API_KEY_ID/KALSHI_PRIVATE_KEY_BASE64 in your environment."
    )


# Batch fetching utilities for parallel execution


async def batch_get_markets(
    client: KalshiSdkAdapter,
    series_tickers: List[str],
    status: Optional[str] = None,
    limit: int = 200,
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Fetch markets for multiple series tickers in parallel.

    Parameters
    ----------
    client : KalshiSdkAdapter
        Kalshi client (must be KalshiSdkAdapter for async support)
    series_tickers : List[str]
        List of series tickers to fetch
    status : Optional[str]
        Filter by market status
    limit : int
        Max markets per series

    Returns
    -------
    Dict[str, List[Dict[str, Any]]]
        Mapping of series_ticker -> list of markets
    """
    if not isinstance(client, KalshiSdkAdapter):
        raise TypeError("batch_get_markets requires KalshiSdkAdapter client")

    # Create tasks for parallel execution
    tasks = [
        client.get_markets_async(
            series_ticker=ticker,
            status=status,
            limit=limit,
        )
        for ticker in series_tickers
    ]

    # Execute all requests in parallel
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Build result mapping
    markets_by_series = {}
    for ticker, result in zip(series_tickers, results):
        if isinstance(result, Exception):
            # Log error but continue
            print(f"Error fetching {ticker}: {result}")
            markets_by_series[ticker] = []
        else:
            markets_by_series[ticker] = result

    return markets_by_series


async def batch_get_series(
    client: KalshiSdkAdapter,
    series_tickers: List[str],
) -> Dict[str, Dict[str, Any]]:
    """
    Fetch series info for multiple tickers in parallel.

    Parameters
    ----------
    client : KalshiSdkAdapter
        Kalshi client (must be KalshiSdkAdapter for async support)
    series_tickers : List[str]
        List of series tickers to fetch

    Returns
    -------
    Dict[str, Dict[str, Any]]
        Mapping of series_ticker -> series info
    """
    if not isinstance(client, KalshiSdkAdapter):
        raise TypeError("batch_get_series requires KalshiSdkAdapter client")

    tasks = [client.get_series_async(ticker) for ticker in series_tickers]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    series_by_ticker = {}
    for ticker, result in zip(series_tickers, results):
        if isinstance(result, Exception):
            print(f"Error fetching series {ticker}: {result}")
            series_by_ticker[ticker] = {}
        else:
            series_by_ticker[ticker] = result

    return series_by_ticker


async def batch_get_events(
    client: KalshiSdkAdapter,
    event_tickers: List[str],
    with_nested_markets: bool = True,
) -> Dict[str, Dict[str, Any]]:
    """
    Fetch events for multiple tickers in parallel.

    Parameters
    ----------
    client : KalshiSdkAdapter
        Kalshi client (must be KalshiSdkAdapter for async support)
    event_tickers : List[str]
        List of event tickers to fetch
    with_nested_markets : bool
        Include nested markets in response

    Returns
    -------
    Dict[str, Dict[str, Any]]
        Mapping of event_ticker -> event info
    """
    if not isinstance(client, KalshiSdkAdapter):
        raise TypeError("batch_get_events requires KalshiSdkAdapter client")

    tasks = [
        client.get_event_async(ticker, with_nested_markets=with_nested_markets)
        for ticker in event_tickers
    ]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    events_by_ticker = {}
    for ticker, result in zip(event_tickers, results):
        if isinstance(result, Exception):
            print(f"Error fetching event {ticker}: {result}")
            events_by_ticker[ticker] = {"event": {}}
        else:
            events_by_ticker[ticker] = result

    return events_by_ticker
