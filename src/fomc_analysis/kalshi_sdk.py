"""
Utilities for connecting to Kalshi using the official Python SDK.

The SDK requires an RSA private key. This module supports loading that
key from a file generated from the base64-encoded key stored in
KALSHI_PRIVATE_KEY_BASE64.
"""

from __future__ import annotations

import asyncio
import base64
import os
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, Optional

from kalshi_python_async import Configuration, KalshiClient as KalshiSdkClient
from kalshi_python_async.models.exchange_status import ExchangeStatus

from .config import settings

PRIVATE_KEY_FILENAME = "kalshi_private_key.pem"


def _decode_private_key(private_key_base64: str) -> bytes:
    try:
        return base64.b64decode(private_key_base64, validate=True)
    except (ValueError, TypeError) as exc:
        raise ValueError("Invalid base64-encoded Kalshi private key.") from exc


@contextmanager
def temporary_private_key_file(private_key_base64: str) -> Iterator[Path]:
    """Write the private key to a temporary file and yield the path."""
    private_key_bytes = _decode_private_key(private_key_base64)
    with tempfile.TemporaryDirectory(prefix="kalshi-key-") as tmp_dir:
        key_path = Path(tmp_dir) / PRIVATE_KEY_FILENAME
        key_path.write_bytes(private_key_bytes)
        os.chmod(key_path, 0o600)
        yield key_path


def load_private_key_pem_from_file(path: Path) -> str:
    """Load the private key PEM contents from a file."""
    return path.read_text(encoding="utf-8")


def create_kalshi_sdk_client(
    *,
    api_key_id: Optional[str] = None,
    private_key_base64: Optional[str] = None,
    base_url: Optional[str] = None,
) -> KalshiSdkClient:
    """Create a Kalshi SDK client using a temporary key file."""
    api_key_id = api_key_id or settings.kalshi_api_key_id
    private_key_base64 = private_key_base64 or settings.kalshi_private_key_base64
    base_url = base_url or settings.kalshi_base_url

    if not api_key_id or not private_key_base64:
        raise ValueError(
            "Kalshi SDK credentials are required. "
            "Set KALSHI_API_KEY_ID and KALSHI_PRIVATE_KEY_BASE64."
        )

    with temporary_private_key_file(private_key_base64) as key_path:
        private_key_pem = load_private_key_pem_from_file(key_path)

    configuration = Configuration(host=base_url)
    configuration.api_key_id = api_key_id
    configuration.private_key_pem = private_key_pem

    return KalshiSdkClient(configuration=configuration)


async def fetch_exchange_status(client: KalshiSdkClient) -> ExchangeStatus:
    """Fetch the exchange status as a connectivity check."""
    return await client.get_exchange_status()


def verify_kalshi_connection() -> ExchangeStatus:
    """Build a Kalshi client from env vars and verify connectivity."""
    client = create_kalshi_sdk_client()
    return asyncio.run(fetch_exchange_status(client))


if __name__ == "__main__":
    status = verify_kalshi_connection()
    print(
        "Kalshi exchange status:",
        {
            "exchange_status": status.exchange_status,
            "exchange_time": status.exchange_time,
        },
    )
