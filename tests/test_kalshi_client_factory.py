"""Tests for Kalshi client factory fallback logic."""

from __future__ import annotations

import types

import pytest

from fomc_analysis import kalshi_client_factory as factory


class DummyClient:
    def __init__(self):
        self.closed = False

    def close(self):
        self.closed = True


def test_get_kalshi_client_prefers_legacy(monkeypatch):
    """Legacy credentials should be used when both key+secret are available."""

    monkeypatch.setenv("KALSHI_API_KEY", "abc")
    monkeypatch.setenv("KALSHI_API_SECRET", "xyz")
    monkeypatch.setattr(
        factory,
        "LegacyKalshiClient",
        lambda: DummyClient(),
    )
    # Ensure SDK path is not triggered even if configured
    monkeypatch.setattr(factory, "has_sdk_credentials", lambda: True)

    client = factory.get_kalshi_client()
    assert isinstance(client, DummyClient)

    # Clean up env manipulations for other tests
    monkeypatch.delenv("KALSHI_API_KEY")
    monkeypatch.delenv("KALSHI_API_SECRET")


def test_get_kalshi_client_sdk(monkeypatch):
    """SDK credentials should be used when RSA pair is configured."""

    dummy_client = DummyClient()
    monkeypatch.setattr(factory, "KalshiSdkAdapter", lambda: dummy_client)

    # Remove legacy env if previously set
    monkeypatch.delenv("KALSHI_API_KEY", raising=False)
    monkeypatch.delenv("KALSHI_API_SECRET", raising=False)

    # Provide SDK credentials via settings object
    monkeypatch.setattr(factory.settings, "kalshi_api_key_id", "key-id", raising=False)
    monkeypatch.setattr(
        factory.settings, "kalshi_private_key_base64", "cHJpdmF0ZS1rZXk=", raising=False
    )

    client = factory.get_kalshi_client()
    assert client is dummy_client

    # reset settings to avoid leaking state
    monkeypatch.setattr(factory.settings, "kalshi_api_key_id", None, raising=False)
    monkeypatch.setattr(factory.settings, "kalshi_private_key_base64", None, raising=False)


def test_get_kalshi_client_missing_creds(monkeypatch):
    """Missing credentials should raise ValueError."""

    monkeypatch.delenv("KALSHI_API_KEY", raising=False)
    monkeypatch.delenv("KALSHI_API_SECRET", raising=False)
    monkeypatch.setattr(factory.settings, "kalshi_api_key_id", None, raising=False)
    monkeypatch.setattr(factory.settings, "kalshi_private_key_base64", None, raising=False)

    with pytest.raises(ValueError):
        factory.get_kalshi_client()


def test_sdk_adapter_close(monkeypatch):
    class DummySDK:
        def __init__(self):
            self.closed = False

        async def close(self):
            self.closed = True

    class DummyResponse:
        def __init__(self, payload):
            self.payload = payload
            self.released = False

        async def json(self):
            return self.payload

        async def read(self):
            return b"{}"

        def release(self):
            self.released = True

    class DummyMarketApi:
        def __init__(self, client):
            self.client = client

        async def get_markets_without_preload_content(self, **kwargs):
            return DummyResponse({"markets": []})

        async def get_series(self, **kwargs):
            class Response:
                def to_dict(self_inner):
                    return {}
            return Response()

    class DummyEventsApi:
        def __init__(self, client):
            self.client = client

        async def get_event_without_preload_content(self, **kwargs):
            return DummyResponse({"event": {}})

    dummy_sdk = DummySDK()
    monkeypatch.setattr(factory, "create_kalshi_sdk_client", lambda: dummy_sdk)
    monkeypatch.setattr(factory, "MarketApi", DummyMarketApi)
    monkeypatch.setattr(factory, "EventsApi", DummyEventsApi)

    adapter = factory.KalshiSdkAdapter()
    adapter.get_markets()
    adapter.get_event("foo")
    adapter.get_series("bar")
    adapter.close()
    assert dummy_sdk.closed
