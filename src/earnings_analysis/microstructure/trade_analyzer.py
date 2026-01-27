"""
DuckDB-powered analytical queries on trade and market data.

Runs SQL-on-Parquet analytical queries against the microstructure
dataset, replicating key analyses from Becker (2025) but focused
specifically on Kalshi earnings mention contracts.

Key analyses:
- Win rate by price level (calibration curve)
- Maker vs taker returns (if taker_side available)
- Spread dynamics over time
- Volume patterns
- Market efficiency scoring
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

try:
    import duckdb
    HAS_DUCKDB = True
except ImportError:
    HAS_DUCKDB = False

from .trade_storage import ParquetStorage


def _require_duckdb():
    if not HAS_DUCKDB:
        raise ImportError(
            "duckdb is required for trade analysis. "
            "Install with: uv pip install duckdb"
        )


class EarningsTradeAnalyzer:
    """
    Analytical engine for earnings mention contract microstructure data.

    Uses DuckDB for fast SQL queries on Parquet files. All queries are
    read-only and produce DataFrames suitable for visualization or
    further analysis.

    Parameters
    ----------
    storage : ParquetStorage
        Data storage containing trades and market data.
    """

    def __init__(self, storage: Optional[ParquetStorage] = None):
        _require_duckdb()
        self.storage = storage or ParquetStorage()
        self.conn = duckdb.connect()

    def close(self):
        """Close the DuckDB connection."""
        self.conn.close()

    def _trades_glob(self, series_ticker: str) -> str:
        """Get glob pattern for trade Parquet files."""
        return str(self.storage.data_dir / "trades" / series_ticker / "chunk_*.parquet")

    def _markets_path(self, series_ticker: str) -> str:
        """Get path to markets Parquet file."""
        return str(self.storage.data_dir / "markets" / f"{series_ticker}_markets.parquet")

    # ── Calibration Analysis ──────────────────────────

    def win_rate_by_price(
        self,
        series_ticker: str,
    ) -> pd.DataFrame:
        """
        Compute actual win rate at each price level for finalized markets.

        This is the earnings-specific calibration curve. Compare with the
        aggregate calibration from Becker (2025) to determine if earnings
        contracts behave like "Finance" (efficient) or "Entertainment"
        (inefficient).

        Parameters
        ----------
        series_ticker : str
            e.g., "KXEARNINGSMENTIONMETA"

        Returns
        -------
        pd.DataFrame
            Columns: price_cents, trade_count, actual_win_rate,
            implied_probability, mispricing
        """
        markets_path = self._markets_path(series_ticker)

        return self.conn.execute(f"""
            WITH finalized AS (
                SELECT ticker, last_price, result
                FROM read_parquet('{markets_path}')
                WHERE status = 'finalized'
                  AND result IN ('yes', 'no')
                  AND last_price > 0
            )
            SELECT
                last_price AS price_cents,
                COUNT(*) AS market_count,
                AVG(CASE WHEN result = 'yes' THEN 1 ELSE 0 END) AS actual_win_rate,
                last_price / 100.0 AS implied_probability,
                AVG(CASE WHEN result = 'yes' THEN 1 ELSE 0 END) - last_price / 100.0 AS mispricing
            FROM finalized
            GROUP BY last_price
            ORDER BY last_price
        """).fetchdf()

    def spread_analysis(
        self,
        series_ticker: str,
    ) -> pd.DataFrame:
        """
        Analyze bid-ask spreads across all markets for a series.

        Parameters
        ----------
        series_ticker : str
            e.g., "KXEARNINGSMENTIONMETA"

        Returns
        -------
        pd.DataFrame
            Spread statistics by status, including avg, median, max spread.
        """
        markets_path = self._markets_path(series_ticker)

        return self.conn.execute(f"""
            SELECT
                status,
                COUNT(*) AS market_count,
                AVG(yes_ask - yes_bid) AS avg_spread_cents,
                MEDIAN(yes_ask - yes_bid) AS median_spread_cents,
                MAX(yes_ask - yes_bid) AS max_spread_cents,
                MIN(yes_ask - yes_bid) AS min_spread_cents,
                AVG(volume) AS avg_volume,
                AVG(open_interest) AS avg_open_interest
            FROM read_parquet('{markets_path}')
            WHERE yes_ask > 0 AND yes_bid >= 0
            GROUP BY status
            ORDER BY market_count DESC
        """).fetchdf()

    def volume_by_word(
        self,
        series_ticker: str,
    ) -> pd.DataFrame:
        """
        Rank words by trading volume.

        High-volume words attract more market maker attention and tend
        to be more efficiently priced.

        Parameters
        ----------
        series_ticker : str
            e.g., "KXEARNINGSMENTIONMETA"

        Returns
        -------
        pd.DataFrame
            Columns: word, total_volume, market_count, avg_last_price, status
        """
        markets_path = self._markets_path(series_ticker)

        return self.conn.execute(f"""
            SELECT
                custom_strike_word AS word,
                SUM(volume) AS total_volume,
                COUNT(*) AS market_count,
                AVG(last_price) AS avg_last_price,
                MAX(status) AS latest_status
            FROM read_parquet('{markets_path}')
            WHERE custom_strike_word != ''
            GROUP BY custom_strike_word
            ORDER BY total_volume DESC
        """).fetchdf()

    def outcome_distribution(
        self,
        series_ticker: str,
    ) -> pd.DataFrame:
        """
        Analyze outcome distribution for finalized markets.

        Returns
        -------
        pd.DataFrame
            YES/NO outcome counts and rates by word.
        """
        markets_path = self._markets_path(series_ticker)

        return self.conn.execute(f"""
            SELECT
                custom_strike_word AS word,
                COUNT(*) AS total_markets,
                SUM(CASE WHEN result = 'yes' THEN 1 ELSE 0 END) AS yes_outcomes,
                SUM(CASE WHEN result = 'no' THEN 1 ELSE 0 END) AS no_outcomes,
                AVG(CASE WHEN result = 'yes' THEN 1 ELSE 0 END) AS yes_rate,
                AVG(last_price) AS avg_last_price
            FROM read_parquet('{markets_path}')
            WHERE status = 'finalized'
              AND result IN ('yes', 'no')
              AND custom_strike_word != ''
            GROUP BY custom_strike_word
            ORDER BY yes_rate DESC
        """).fetchdf()

    # ── Taker-Side Analysis (requires trade data) ──────

    def taker_side_distribution(
        self,
        series_ticker: str,
    ) -> pd.DataFrame:
        """
        Analyze taker side distribution (YES vs NO aggression).

        Becker found takers disproportionately buy YES. This checks
        if the same pattern holds for earnings mention contracts.

        Parameters
        ----------
        series_ticker : str
            Series ticker.

        Returns
        -------
        pd.DataFrame
            Taker side counts and percentages.
        """
        trades_glob = self._trades_glob(series_ticker)

        return self.conn.execute(f"""
            SELECT
                taker_side,
                COUNT(*) AS trade_count,
                SUM(count) AS total_contracts,
                AVG(yes_price) AS avg_price_cents,
                COUNT(*) * 100.0 / SUM(COUNT(*)) OVER () AS pct_of_trades
            FROM read_parquet('{trades_glob}')
            GROUP BY taker_side
            ORDER BY trade_count DESC
        """).fetchdf()

    def taker_returns_by_price(
        self,
        series_ticker: str,
    ) -> pd.DataFrame:
        """
        Compute taker returns by price bucket.

        Joins trades with market outcomes to compute actual returns
        for takers at each price level.

        Parameters
        ----------
        series_ticker : str
            Series ticker.

        Returns
        -------
        pd.DataFrame
            Taker returns by price decile.
        """
        trades_glob = self._trades_glob(series_ticker)
        markets_path = self._markets_path(series_ticker)

        return self.conn.execute(f"""
            WITH trade_outcomes AS (
                SELECT
                    t.ticker,
                    t.yes_price,
                    t.taker_side,
                    t.count AS contracts,
                    m.result,
                    CASE
                        WHEN t.taker_side = 'yes' AND m.result = 'yes'
                            THEN (100 - t.yes_price) * t.count
                        WHEN t.taker_side = 'yes' AND m.result = 'no'
                            THEN -t.yes_price * t.count
                        WHEN t.taker_side = 'no' AND m.result = 'no'
                            THEN t.yes_price * t.count
                        WHEN t.taker_side = 'no' AND m.result = 'yes'
                            THEN -(100 - t.yes_price) * t.count
                        ELSE 0
                    END AS pnl_cents
                FROM read_parquet('{trades_glob}') t
                JOIN read_parquet('{markets_path}') m ON t.ticker = m.ticker
                WHERE m.status = 'finalized'
                  AND m.result IN ('yes', 'no')
            )
            SELECT
                FLOOR(yes_price / 10) * 10 AS price_bucket,
                COUNT(*) AS trade_count,
                AVG(pnl_cents) AS avg_pnl_cents,
                SUM(pnl_cents) AS total_pnl_cents,
                AVG(CASE WHEN pnl_cents > 0 THEN 1 ELSE 0 END) AS win_rate
            FROM trade_outcomes
            GROUP BY FLOOR(yes_price / 10) * 10
            ORDER BY price_bucket
        """).fetchdf()

    # ── Cross-Ticker Comparison ────────────────────────

    def cross_ticker_efficiency(
        self,
        companies: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        Compare market efficiency across company tickers.

        Computes efficiency metrics for each company based on:
        - Average spread
        - Volume
        - Calibration deviation

        Parameters
        ----------
        companies : list of str, optional
            Company tickers. Defaults to scanning stored data.

        Returns
        -------
        pd.DataFrame
            Efficiency metrics per company.
        """
        if companies is None:
            # Scan for available data
            tickers = self.storage.list_trade_tickers()
            companies = [
                t.replace("KXEARNINGSMENTION", "")
                for t in tickers
            ]

        if not companies:
            # Try reading from market files
            markets_dir = self.storage.data_dir / "markets"
            if markets_dir.exists():
                companies = [
                    f.stem.replace("KXEARNINGSMENTION", "").replace("_markets", "")
                    for f in markets_dir.glob("KXEARNINGSMENTION*_markets.parquet")
                ]

        if not companies:
            return pd.DataFrame()

        results = []
        for company in companies:
            series_ticker = f"KXEARNINGSMENTION{company}"
            markets_path = self.storage.data_dir / "markets" / f"{series_ticker}_markets.parquet"

            if not markets_path.exists():
                continue

            try:
                row = self.conn.execute(f"""
                    SELECT
                        '{company}' AS company,
                        COUNT(*) AS total_markets,
                        AVG(yes_ask - yes_bid) AS avg_spread_cents,
                        AVG(volume) AS avg_volume,
                        SUM(volume) AS total_volume,
                        AVG(CASE WHEN status = 'finalized' THEN 1 ELSE 0 END) AS finalized_pct,
                        COUNT(CASE WHEN status = 'finalized' THEN 1 END) AS finalized_count
                    FROM read_parquet('{markets_path}')
                    WHERE yes_ask > 0
                """).fetchdf()
                results.append(row)
            except Exception:
                continue

        if not results:
            return pd.DataFrame()

        return pd.concat(results, ignore_index=True)

    # ── Earnings-Specific Calibration Curve ────────────

    def build_earnings_calibration(
        self,
        companies: Optional[List[str]] = None,
    ) -> Dict[int, float]:
        """
        Build an empirical calibration curve from earnings contract outcomes.

        Aggregates across all companies to build price -> win_rate mapping.
        This can be loaded into KalshiCalibrationCurve to replace the
        parametric model with earnings-specific empirical data.

        Parameters
        ----------
        companies : list of str, optional
            Company tickers.

        Returns
        -------
        dict
            Mapping of price_cents (1-99) -> actual_win_rate.
        """
        if companies is None:
            markets_dir = self.storage.data_dir / "markets"
            if not markets_dir.exists():
                return {}
            companies = [
                f.stem.replace("KXEARNINGSMENTION", "").replace("_markets", "")
                for f in markets_dir.glob("KXEARNINGSMENTION*_markets.parquet")
            ]

        if not companies:
            return {}

        # Collect all finalized markets across companies
        all_results = []
        for company in companies:
            series_ticker = f"KXEARNINGSMENTION{company}"
            markets_path = self.storage.data_dir / "markets" / f"{series_ticker}_markets.parquet"
            if not markets_path.exists():
                continue

            try:
                df = self.conn.execute(f"""
                    SELECT last_price, result
                    FROM read_parquet('{markets_path}')
                    WHERE status = 'finalized'
                      AND result IN ('yes', 'no')
                      AND last_price BETWEEN 1 AND 99
                """).fetchdf()
                all_results.append(df)
            except Exception:
                continue

        if not all_results:
            return {}

        combined = pd.concat(all_results, ignore_index=True)
        combined["won"] = (combined["result"] == "yes").astype(int)

        # Group by price level, require minimum 3 observations
        grouped = combined.groupby("last_price").agg(
            win_rate=("won", "mean"),
            count=("won", "count"),
        ).reset_index()

        # Filter to bins with enough data
        reliable = grouped[grouped["count"] >= 3]

        return {
            int(row["last_price"]): float(row["win_rate"])
            for _, row in reliable.iterrows()
        }
