"""
Parquet-based storage for Kalshi trade and market data.

Follows the chunked Parquet approach from Becker (2025):
- Data stored as chunked Parquet files (one per ticker or batch)
- Supports append, deduplication, and bulk read
- Efficient columnar storage with compression

Directory layout:
    data/microstructure/
    ├── markets/
    │   └── *.parquet          # Market metadata (all tickers)
    └── trades/
        ├── KXEARNINGSMENTIONMETA/
        │   ├── chunk_0000.parquet
        │   └── chunk_0001.parquet
        ├── KXEARNINGSMENTIONTSLA/
        │   └── chunk_0000.parquet
        └── ...
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


_DEFAULT_DATA_DIR = Path("data/microstructure")
_CHUNK_SIZE = 10_000


@dataclass
class TradeRecord:
    """A single trade from the Kalshi API."""
    trade_id: str
    ticker: str
    count: int            # Number of contracts
    yes_price: int        # Price in cents
    taker_side: str       # "yes" or "no"
    created_time: str     # ISO 8601 timestamp

    def to_dict(self) -> Dict:
        return {
            "trade_id": self.trade_id,
            "ticker": self.ticker,
            "count": self.count,
            "yes_price": self.yes_price,
            "taker_side": self.taker_side,
            "created_time": self.created_time,
        }


@dataclass
class MarketRecord:
    """Minimal market metadata for microstructure analysis."""
    ticker: str
    series_ticker: str
    title: str
    status: str           # active, settled, finalized, closed
    yes_bid: int          # Best bid in cents
    yes_ask: int          # Best ask in cents
    last_price: int       # Last trade price in cents
    volume: int           # Total volume
    open_interest: int    # Open interest
    result: Optional[str] # "yes" or "no" for settled markets
    expiration_time: str  # ISO 8601
    custom_strike: Optional[Dict] = None

    def to_dict(self) -> Dict:
        return {
            "ticker": self.ticker,
            "series_ticker": self.series_ticker,
            "title": self.title,
            "status": self.status,
            "yes_bid": self.yes_bid,
            "yes_ask": self.yes_ask,
            "last_price": self.last_price,
            "volume": self.volume,
            "open_interest": self.open_interest,
            "result": self.result,
            "expiration_time": self.expiration_time,
            "custom_strike_word": (
                self.custom_strike.get("Word", "")
                if self.custom_strike else ""
            ),
        }


class ParquetStorage:
    """
    Chunked Parquet storage for trade and market data.

    Parameters
    ----------
    data_dir : Path
        Root directory for Parquet files.
    chunk_size : int
        Max records per chunk file (default: 10,000).
    """

    def __init__(
        self,
        data_dir: Path = _DEFAULT_DATA_DIR,
        chunk_size: int = _CHUNK_SIZE,
    ):
        self.data_dir = Path(data_dir)
        self.chunk_size = chunk_size
        self.data_dir.mkdir(parents=True, exist_ok=True)

    # ── Trade storage ──────────────────────────────────

    def _trades_dir(self, series_ticker: str) -> Path:
        d = self.data_dir / "trades" / series_ticker
        d.mkdir(parents=True, exist_ok=True)
        return d

    def _next_chunk_path(self, series_ticker: str) -> Path:
        trades_dir = self._trades_dir(series_ticker)
        existing = sorted(trades_dir.glob("chunk_*.parquet"))
        idx = len(existing)
        return trades_dir / f"chunk_{idx:04d}.parquet"

    def save_trades(
        self,
        series_ticker: str,
        trades: List[Dict],
    ) -> int:
        """
        Save trade records to Parquet, chunking if necessary.

        Parameters
        ----------
        series_ticker : str
            Series ticker (e.g., "KXEARNINGSMENTIONMETA").
        trades : list of dict
            Trade records with keys matching TradeRecord fields.

        Returns
        -------
        int
            Number of records saved.
        """
        if not trades:
            return 0

        df = pd.DataFrame(trades)

        # Deduplicate by trade_id
        if "trade_id" in df.columns:
            existing_df = self.read_trades(series_ticker)
            if not existing_df.empty and "trade_id" in existing_df.columns:
                existing_ids = set(existing_df["trade_id"])
                df = df[~df["trade_id"].isin(existing_ids)]

        if df.empty:
            return 0

        # Chunk and save
        saved = 0
        for start in range(0, len(df), self.chunk_size):
            chunk = df.iloc[start:start + self.chunk_size]
            path = self._next_chunk_path(series_ticker)
            table = pa.Table.from_pandas(chunk, preserve_index=False)
            pq.write_table(table, path, compression="snappy")
            saved += len(chunk)

        return saved

    def read_trades(self, series_ticker: str) -> pd.DataFrame:
        """Read all trades for a series ticker."""
        trades_dir = self._trades_dir(series_ticker)
        files = sorted(trades_dir.glob("chunk_*.parquet"))
        if not files:
            return pd.DataFrame()
        dfs = [pq.read_table(f).to_pandas() for f in files]
        return pd.concat(dfs, ignore_index=True)

    def trade_count(self, series_ticker: str) -> int:
        """Count total trades stored for a ticker."""
        df = self.read_trades(series_ticker)
        return len(df)

    def list_trade_tickers(self) -> List[str]:
        """List all series tickers with stored trade data."""
        trades_root = self.data_dir / "trades"
        if not trades_root.exists():
            return []
        return sorted(
            d.name for d in trades_root.iterdir()
            if d.is_dir() and list(d.glob("chunk_*.parquet"))
        )

    # ── Market storage ─────────────────────────────────

    def _markets_dir(self) -> Path:
        d = self.data_dir / "markets"
        d.mkdir(parents=True, exist_ok=True)
        return d

    def save_markets(
        self,
        markets: List[Dict],
        filename: str = "earnings_markets.parquet",
    ) -> int:
        """
        Save market metadata to a single Parquet file.

        Overwrites any existing file (markets are point-in-time snapshots).

        Parameters
        ----------
        markets : list of dict
            Market records.
        filename : str
            Output filename.

        Returns
        -------
        int
            Number of records saved.
        """
        if not markets:
            return 0

        df = pd.DataFrame(markets)
        path = self._markets_dir() / filename
        table = pa.Table.from_pandas(df, preserve_index=False)
        pq.write_table(table, path, compression="snappy")
        return len(df)

    def read_markets(
        self,
        filename: str = "earnings_markets.parquet",
    ) -> pd.DataFrame:
        """Read market metadata."""
        path = self._markets_dir() / filename
        if not path.exists():
            return pd.DataFrame()
        return pq.read_table(path).to_pandas()

    # ── Snapshots (for time-series tracking) ───────────

    def save_snapshot(
        self,
        snapshot_data: List[Dict],
        snapshot_name: str,
    ) -> int:
        """
        Save a timestamped snapshot (e.g., daily market prices).

        Saved to data_dir/snapshots/{snapshot_name}/{date}.parquet.
        """
        if not snapshot_data:
            return 0

        from datetime import date
        today = date.today().isoformat()
        snap_dir = self.data_dir / "snapshots" / snapshot_name
        snap_dir.mkdir(parents=True, exist_ok=True)

        df = pd.DataFrame(snapshot_data)
        path = snap_dir / f"{today}.parquet"
        table = pa.Table.from_pandas(df, preserve_index=False)
        pq.write_table(table, path, compression="snappy")
        return len(df)

    def read_snapshots(
        self,
        snapshot_name: str,
    ) -> pd.DataFrame:
        """Read all snapshots for a given name, sorted by date."""
        snap_dir = self.data_dir / "snapshots" / snapshot_name
        if not snap_dir.exists():
            return pd.DataFrame()
        files = sorted(snap_dir.glob("*.parquet"))
        if not files:
            return pd.DataFrame()
        dfs = []
        for f in files:
            df = pq.read_table(f).to_pandas()
            df["snapshot_date"] = f.stem
            dfs.append(df)
        return pd.concat(dfs, ignore_index=True)

    def data_summary(self) -> Dict:
        """Return summary of all stored data."""
        trade_tickers = self.list_trade_tickers()
        trade_counts = {t: self.trade_count(t) for t in trade_tickers}
        markets_df = self.read_markets()
        return {
            "trade_tickers": trade_tickers,
            "trade_counts": trade_counts,
            "total_trades": sum(trade_counts.values()),
            "markets_count": len(markets_df),
            "data_dir": str(self.data_dir),
        }
