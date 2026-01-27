"""
Regime-aware strategy adaptation for Kalshi prediction market trading.

Monitors market efficiency over time and adapts trading parameters
based on observed regime changes. Key insight from Becker (2025):
the maker-taker dynamic reversed in Q4 2024 when professional market
makers entered. Markets are becoming more efficient over time.

Modules:
- EfficiencyMonitor: Track spread, volume, calibration metrics
- AdaptiveThresholds: Adjust edge thresholds based on efficiency
- NewContractDetector: Find newly listed contracts (higher mispricing)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .trade_storage import ParquetStorage


@dataclass
class EfficiencyMetrics:
    """Market efficiency metrics for a ticker at a point in time."""
    company: str
    date: str
    avg_spread_cents: float       # Lower = more efficient
    avg_volume: float             # Higher = more attention
    spread_narrowing_trend: float # Negative = spreads tightening
    volume_growth_trend: float    # Positive = growing interest
    efficiency_score: float       # 0 (inefficient) to 1 (efficient)
    n_markets: int

    def to_dict(self) -> Dict:
        return {
            "company": self.company,
            "date": self.date,
            "avg_spread_cents": self.avg_spread_cents,
            "avg_volume": self.avg_volume,
            "spread_narrowing_trend": self.spread_narrowing_trend,
            "volume_growth_trend": self.volume_growth_trend,
            "efficiency_score": self.efficiency_score,
            "n_markets": self.n_markets,
        }


class EfficiencyMonitor:
    """
    Monitor market efficiency for earnings mention contracts.

    Computes an efficiency score from 0 (inefficient, more edge)
    to 1 (efficient, less edge) based on:
    - Bid-ask spreads (40% weight): Tight spreads = efficient
    - Volume (30% weight): High volume = more attention
    - Calibration deviation (30% weight): Low deviation = efficient

    Parameters
    ----------
    storage : ParquetStorage
        Data storage with market snapshots.
    spread_efficient_threshold : float
        Spread (cents) at which market is considered efficient.
    volume_efficient_threshold : float
        Average volume at which market is considered efficient.
    """

    def __init__(
        self,
        storage: Optional[ParquetStorage] = None,
        spread_efficient_threshold: float = 3.0,
        volume_efficient_threshold: float = 500.0,
    ):
        self.storage = storage or ParquetStorage()
        self.spread_threshold = spread_efficient_threshold
        self.volume_threshold = volume_efficient_threshold

    def compute_efficiency(
        self,
        company: str,
        markets_df: Optional[pd.DataFrame] = None,
    ) -> EfficiencyMetrics:
        """
        Compute current efficiency score for a company's contracts.

        Parameters
        ----------
        company : str
            Company ticker (e.g., "META").
        markets_df : pd.DataFrame, optional
            Market data. If not provided, reads from storage.

        Returns
        -------
        EfficiencyMetrics
            Current efficiency assessment.
        """
        if markets_df is None:
            series_ticker = f"KXEARNINGSMENTION{company}"
            markets_df = self.storage.read_markets(
                filename=f"{series_ticker}_markets.parquet"
            )

        if markets_df.empty:
            return EfficiencyMetrics(
                company=company,
                date=date.today().isoformat(),
                avg_spread_cents=0,
                avg_volume=0,
                spread_narrowing_trend=0,
                volume_growth_trend=0,
                efficiency_score=0.5,
                n_markets=0,
            )

        # Filter to active markets with valid bid/ask
        active = markets_df
        if "status" in active.columns:
            active = active[active["status"] == "active"]
        if "yes_ask" in active.columns and "yes_bid" in active.columns:
            active = active[(active["yes_ask"] > 0) & (active["yes_bid"] >= 0)]

        n_markets = len(active)
        if n_markets == 0:
            return EfficiencyMetrics(
                company=company,
                date=date.today().isoformat(),
                avg_spread_cents=0,
                avg_volume=0,
                spread_narrowing_trend=0,
                volume_growth_trend=0,
                efficiency_score=0.5,
                n_markets=0,
            )

        # Compute metrics
        spreads = active["yes_ask"] - active["yes_bid"]
        avg_spread = float(spreads.mean())
        avg_volume = float(active["volume"].mean()) if "volume" in active.columns else 0

        # Spread component: 0 if spread > 10c, 1 if spread <= threshold
        spread_score = 1.0 - min(avg_spread / 10.0, 1.0)

        # Volume component: 0 if no volume, 1 if >= threshold
        volume_score = min(avg_volume / self.volume_threshold, 1.0)

        # Combined efficiency score
        efficiency = 0.5 * spread_score + 0.5 * volume_score

        # Trends from snapshots
        spread_trend, volume_trend = self._compute_trends(company)

        return EfficiencyMetrics(
            company=company,
            date=date.today().isoformat(),
            avg_spread_cents=avg_spread,
            avg_volume=avg_volume,
            spread_narrowing_trend=spread_trend,
            volume_growth_trend=volume_trend,
            efficiency_score=float(np.clip(efficiency, 0, 1)),
            n_markets=n_markets,
        )

    def _compute_trends(self, company: str) -> Tuple[float, float]:
        """Compute spread and volume trends from historical snapshots."""
        snapshots = self.storage.read_snapshots("market_prices")
        if snapshots.empty or "company" not in snapshots.columns:
            return 0.0, 0.0

        company_data = snapshots[snapshots["company"] == company]
        if len(company_data) < 2:
            return 0.0, 0.0

        # Group by snapshot date
        by_date = company_data.groupby("snapshot_date").agg(
            avg_spread=("yes_ask", "mean"),
            avg_volume=("volume", "mean"),
        ).sort_index()

        if "yes_bid" in company_data.columns:
            by_date["avg_spread"] = company_data.groupby("snapshot_date").apply(
                lambda g: (g["yes_ask"] - g["yes_bid"]).mean()
            ).values

        if len(by_date) < 2:
            return 0.0, 0.0

        # Simple linear trend (slope)
        x = np.arange(len(by_date))
        spread_trend = float(np.polyfit(x, by_date["avg_spread"].values, 1)[0])
        volume_trend = float(np.polyfit(x, by_date["avg_volume"].values, 1)[0])

        return spread_trend, volume_trend

    def monitor_all(
        self,
        companies: Optional[List[str]] = None,
    ) -> List[EfficiencyMetrics]:
        """
        Compute efficiency for all companies.

        Parameters
        ----------
        companies : list of str, optional
            Company tickers.

        Returns
        -------
        list of EfficiencyMetrics
            One entry per company, sorted by efficiency score (ascending).
        """
        if companies is None:
            companies = ["META", "TSLA", "NVDA", "AAPL", "MSFT", "AMZN", "GOOGL", "NFLX"]

        results = []
        for company in companies:
            metrics = self.compute_efficiency(company)
            if metrics.n_markets > 0:
                results.append(metrics)

        # Sort by efficiency (least efficient = most edge first)
        results.sort(key=lambda m: m.efficiency_score)
        return results


class AdaptiveThresholds:
    """
    Dynamically adjust trading edge thresholds based on market efficiency.

    In efficient markets (tight spreads, high volume), require larger edge.
    In inefficient markets (wide spreads, low volume), trade with smaller edge.

    Parameters
    ----------
    base_yes_threshold : float
        Base YES edge threshold (default: 0.15).
    base_no_threshold : float
        Base NO edge threshold (default: 0.08).
    efficiency_sensitivity : float
        How much to adjust thresholds per unit of efficiency (default: 0.10).
        At efficiency=1.0, thresholds increase by this amount.
        At efficiency=0.0, thresholds decrease by this amount.
    min_threshold : float
        Minimum edge threshold (floor) (default: 0.03).
    max_threshold : float
        Maximum edge threshold (ceiling) (default: 0.30).
    """

    def __init__(
        self,
        base_yes_threshold: float = 0.15,
        base_no_threshold: float = 0.08,
        efficiency_sensitivity: float = 0.10,
        min_threshold: float = 0.03,
        max_threshold: float = 0.30,
    ):
        self.base_yes = base_yes_threshold
        self.base_no = base_no_threshold
        self.sensitivity = efficiency_sensitivity
        self.min_threshold = min_threshold
        self.max_threshold = max_threshold

    def compute_thresholds(
        self,
        efficiency_score: float,
    ) -> Dict[str, float]:
        """
        Compute adaptive thresholds for a given efficiency level.

        Parameters
        ----------
        efficiency_score : float
            Market efficiency from 0 (inefficient) to 1 (efficient).

        Returns
        -------
        dict
            {"yes_threshold": float, "no_threshold": float, "edge_threshold": float}
        """
        # Adjustment: positive when efficient (raise thresholds),
        # negative when inefficient (lower thresholds)
        adjustment = (efficiency_score - 0.5) * 2 * self.sensitivity

        yes_thresh = np.clip(
            self.base_yes + adjustment,
            self.min_threshold,
            self.max_threshold,
        )
        no_thresh = np.clip(
            self.base_no + adjustment,
            self.min_threshold,
            self.max_threshold,
        )
        # General threshold is average of YES/NO
        edge_thresh = (yes_thresh + no_thresh) / 2

        return {
            "yes_threshold": float(yes_thresh),
            "no_threshold": float(no_thresh),
            "edge_threshold": float(edge_thresh),
            "efficiency_score": float(efficiency_score),
            "adjustment": float(adjustment),
        }

    def compute_all_thresholds(
        self,
        efficiency_metrics: List[EfficiencyMetrics],
    ) -> Dict[str, Dict[str, float]]:
        """
        Compute thresholds for all companies.

        Returns
        -------
        dict
            Mapping of company -> threshold dict.
        """
        return {
            m.company: self.compute_thresholds(m.efficiency_score)
            for m in efficiency_metrics
        }


class NewContractDetector:
    """
    Detect newly listed contracts that may be less efficiently priced.

    New contracts tend to have:
    - Wider spreads (less market maker attention)
    - Lower volume (less price discovery)
    - More mispricing (higher edge potential)

    Parameters
    ----------
    storage : ParquetStorage
        Data storage with market snapshots.
    max_age_days : int
        Consider contracts listed within this many days as "new" (default: 30).
    min_spread_cents : int
        Minimum spread to consider a contract as "interesting" (default: 3).
    """

    def __init__(
        self,
        storage: Optional[ParquetStorage] = None,
        max_age_days: int = 30,
        min_spread_cents: int = 3,
    ):
        self.storage = storage or ParquetStorage()
        self.max_age_days = max_age_days
        self.min_spread_cents = min_spread_cents

    def find_new_contracts(
        self,
        companies: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        Find recently listed contracts with wide spreads.

        Parameters
        ----------
        companies : list of str, optional
            Company tickers to scan.

        Returns
        -------
        pd.DataFrame
            New contracts sorted by spread (widest first).
            Columns: ticker, company, word, spread, volume, last_price,
            days_old, opportunity_score.
        """
        if companies is None:
            companies = ["META", "TSLA", "NVDA", "AAPL", "MSFT", "AMZN", "GOOGL", "NFLX"]

        records = []
        now = datetime.utcnow()

        for company in companies:
            series_ticker = f"KXEARNINGSMENTION{company}"
            markets_df = self.storage.read_markets(
                filename=f"{series_ticker}_markets.parquet"
            )

            if markets_df.empty:
                continue

            # Filter to active markets
            if "status" in markets_df.columns:
                active = markets_df[markets_df["status"] == "active"].copy()
            else:
                active = markets_df.copy()

            if active.empty:
                continue

            for _, row in active.iterrows():
                spread = row.get("yes_ask", 0) - row.get("yes_bid", 0)
                if spread < self.min_spread_cents:
                    continue

                # Estimate age from expiration (rough heuristic)
                exp_str = row.get("expiration_time", "")
                days_old = None
                if exp_str:
                    try:
                        exp_dt = datetime.fromisoformat(exp_str.replace("Z", "+00:00"))
                        days_to_exp = (exp_dt.replace(tzinfo=None) - now).days
                        # Contracts typically listed 90 days before expiry
                        days_old = max(0, 90 - days_to_exp)
                    except (ValueError, TypeError):
                        pass

                volume = row.get("volume", 0)
                last_price = row.get("last_price", 50)

                # Opportunity score: high spread + low volume + mid-range price
                price_mid_distance = abs(last_price - 50) / 50  # 0 at 50c, 1 at extremes
                opp_score = (
                    0.4 * min(spread / 10, 1.0) +  # Spread component
                    0.3 * max(0, 1 - volume / 200) +  # Low volume bonus
                    0.3 * (1 - price_mid_distance)  # Mid-range price bonus
                )

                records.append({
                    "ticker": row.get("ticker", ""),
                    "company": company,
                    "word": row.get("custom_strike_word", ""),
                    "spread_cents": spread,
                    "volume": volume,
                    "last_price": last_price,
                    "yes_bid": row.get("yes_bid", 0),
                    "yes_ask": row.get("yes_ask", 0),
                    "days_old": days_old,
                    "opportunity_score": round(opp_score, 3),
                })

        if not records:
            return pd.DataFrame()

        df = pd.DataFrame(records)
        return df.sort_values("opportunity_score", ascending=False).reset_index(drop=True)

    def summarize_opportunities(
        self,
        companies: Optional[List[str]] = None,
    ) -> Dict:
        """
        Summarize new contract opportunities across all companies.

        Returns
        -------
        dict
            Summary with top opportunities and aggregated stats.
        """
        df = self.find_new_contracts(companies)
        if df.empty:
            return {"total_opportunities": 0, "top_contracts": []}

        return {
            "total_opportunities": len(df),
            "by_company": df.groupby("company").size().to_dict(),
            "avg_spread": float(df["spread_cents"].mean()),
            "avg_opportunity_score": float(df["opportunity_score"].mean()),
            "top_contracts": df.head(10).to_dict("records"),
        }
