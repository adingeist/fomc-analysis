"""
Build ground truth dataset from finalized Kalshi earnings mention contracts.

This module fetches all settled/finalized contracts from the Kalshi API and
constructs a structured dataset of actual outcomes: which words were mentioned
during which earnings calls, along with the market prices at settlement.

The resulting dataset is the foundation for real (non-mock) backtesting.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, asdict, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from fomc_analysis.kalshi_client_factory import (
    KalshiClientProtocol,
    KalshiSdkAdapter,
    get_kalshi_client,
)


# Known series tickers for earnings mention contracts
EARNINGS_SERIES_TICKERS = [
    "KXEARNINGSMENTIONMETA",
    "KXEARNINGSMENTIONTSLA",
    "KXEARNINGSMENTIONNVDA",
    "KXEARNINGSMENTIONAMZN",
    "KXEARNINGSMENTIONAAPL",
    "KXEARNINGSMENTIONMSFT",
    "KXEARNINGSMENTIONNFLX",
    "KXEARNINGMENTIONMETA",  # alternate spelling variants
    "KXEARNINGMENTIONTSLA",
    "KXEARNINGMENTIONNVDA",
]


@dataclass
class SettledContract:
    """A single settled Kalshi earnings mention contract."""

    market_ticker: str
    series_ticker: str
    company_ticker: str  # e.g. META, TSLA, NVDA
    word: str  # The word/phrase tracked
    outcome: int  # 1 = YES (mentioned), 0 = NO (not mentioned)
    settlement_price: float  # Final price (0-1 scale)
    event_date: Optional[str] = None  # Extracted from ticker, e.g. "2025-10-30"
    expiration_time: Optional[str] = None
    open_price: Optional[float] = None  # Market price when contract opened
    last_trade_price: Optional[float] = None
    yes_bid: Optional[float] = None
    yes_ask: Optional[float] = None
    volume: Optional[int] = None
    result: Optional[str] = None  # Kalshi "result" field if present

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class GroundTruthDataset:
    """Complete ground truth dataset across all tickers and dates."""

    contracts: List[SettledContract]
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def outcomes_df(self) -> pd.DataFrame:
        """Build outcomes DataFrame: index=event_date, columns=word, values=0/1.

        Groups by company ticker and event date, pivoting words into columns.
        Returns one DataFrame per company or a combined one with multi-level columns.
        """
        if not self.contracts:
            return pd.DataFrame()

        rows = []
        for c in self.contracts:
            rows.append({
                "company": c.company_ticker,
                "event_date": c.event_date or "unknown",
                "word": c.word,
                "outcome": c.outcome,
            })

        df = pd.DataFrame(rows)
        if df.empty:
            return df

        # Pivot: rows = (company, event_date), columns = word, values = outcome
        pivoted = df.pivot_table(
            index=["company", "event_date"],
            columns="word",
            values="outcome",
            aggfunc="first",
        )
        return pivoted

    @property
    def market_prices_df(self) -> pd.DataFrame:
        """Build market prices DataFrame from settlement prices."""
        if not self.contracts:
            return pd.DataFrame()

        rows = []
        for c in self.contracts:
            rows.append({
                "company": c.company_ticker,
                "event_date": c.event_date or "unknown",
                "word": c.word,
                "market_price": c.settlement_price,
            })

        df = pd.DataFrame(rows)
        if df.empty:
            return df

        pivoted = df.pivot_table(
            index=["company", "event_date"],
            columns="word",
            values="market_price",
            aggfunc="first",
        )
        return pivoted

    def for_ticker(self, company_ticker: str) -> "GroundTruthDataset":
        """Filter dataset to a single company ticker."""
        filtered = [c for c in self.contracts if c.company_ticker == company_ticker]
        return GroundTruthDataset(contracts=filtered, metadata=self.metadata)

    @property
    def companies(self) -> List[str]:
        """List unique company tickers in dataset."""
        return sorted(set(c.company_ticker for c in self.contracts))

    @property
    def event_dates(self) -> List[str]:
        """List unique event dates in dataset."""
        return sorted(set(c.event_date for c in self.contracts if c.event_date))

    def summary(self) -> Dict[str, Any]:
        """Generate summary statistics."""
        if not self.contracts:
            return {"total_contracts": 0}

        companies = self.companies
        yes_count = sum(1 for c in self.contracts if c.outcome == 1)
        no_count = sum(1 for c in self.contracts if c.outcome == 0)

        return {
            "total_contracts": len(self.contracts),
            "companies": companies,
            "num_companies": len(companies),
            "event_dates": self.event_dates,
            "num_event_dates": len(self.event_dates),
            "yes_outcomes": yes_count,
            "no_outcomes": no_count,
            "yes_rate": yes_count / len(self.contracts) if self.contracts else 0,
            "unique_words": sorted(set(c.word for c in self.contracts)),
        }


def _extract_company_from_series(series_ticker: str) -> str:
    """Extract company ticker from series ticker.

    e.g. KXEARNINGSMENTIONMETA -> META
    """
    for prefix in ["KXEARNINGSMENTION", "KXEARNINGMENTION"]:
        if series_ticker.upper().startswith(prefix):
            return series_ticker[len(prefix):].upper()
    return series_ticker.upper()


def _extract_event_date_from_ticker(market_ticker: str) -> Optional[str]:
    """Extract event date from market ticker.

    Kalshi tickers like: KXEARNINGSMENTIONMETA-26JAN29-AI
    The middle segment is a date code: 26JAN29 = 2026-01-29

    Returns ISO date string or None.
    """
    parts = market_ticker.split("-")
    if len(parts) < 2:
        return None

    date_part = parts[1]

    # Pattern: YYMMMDD (e.g. 26JAN29, 25OCT30)
    match = re.match(r"(\d{2})([A-Z]{3})(\d{2})", date_part)
    if not match:
        return None

    year_short = int(match.group(1))
    month_str = match.group(2)
    day = int(match.group(3))

    month_map = {
        "JAN": 1, "FEB": 2, "MAR": 3, "APR": 4,
        "MAY": 5, "JUN": 6, "JUL": 7, "AUG": 8,
        "SEP": 9, "OCT": 10, "NOV": 11, "DEC": 12,
    }
    month = month_map.get(month_str)
    if month is None:
        return None

    year = 2000 + year_short
    try:
        dt = datetime(year, month, day)
        return dt.strftime("%Y-%m-%d")
    except ValueError:
        return None


def _parse_settled_market(
    market: Dict[str, Any],
    series_ticker: str,
) -> Optional[SettledContract]:
    """Parse a single settled market into a SettledContract."""
    market_ticker = market.get("ticker", "")
    status = market.get("status", "").lower()

    # Only process settled/finalized contracts
    if status not in ("settled", "finalized", "closed"):
        return None

    # Extract word from custom_strike or yes_sub_title
    custom_strike = market.get("custom_strike") or {}
    word = custom_strike.get("Word") or market.get("yes_sub_title", "")

    if not word:
        # Try to extract from title
        title = market.get("title", "")
        # Pattern: last part after the dash in ticker
        parts = market_ticker.split("-")
        if len(parts) >= 3:
            word = parts[-1]
        else:
            word = title

    # Normalize word
    word = word.strip()
    if not word:
        return None

    # Determine outcome from result or last_price
    result_field = market.get("result", "")
    last_price = market.get("last_price", 0)

    if result_field:
        outcome = 1 if result_field.lower() in ("yes", "true", "1") else 0
    elif last_price is not None:
        # Settlement: last_price near 100 = YES, near 0 = NO
        outcome = 1 if last_price > 50 else 0
    else:
        return None

    # Settlement price (0-1 scale)
    settlement_price = (last_price or 0) / 100.0

    # Extract metadata
    company_ticker = _extract_company_from_series(series_ticker)
    event_date = _extract_event_date_from_ticker(market_ticker)

    return SettledContract(
        market_ticker=market_ticker,
        series_ticker=series_ticker,
        company_ticker=company_ticker,
        word=word,
        outcome=outcome,
        settlement_price=settlement_price,
        event_date=event_date,
        expiration_time=market.get("expiration_time"),
        last_trade_price=(market.get("last_price", 0) or 0) / 100.0,
        yes_bid=(market.get("yes_bid", 0) or 0) / 100.0,
        yes_ask=(market.get("yes_ask", 0) or 0) / 100.0,
        volume=market.get("volume"),
        result=result_field or None,
    )


def fetch_ground_truth(
    client: Optional[KalshiClientProtocol] = None,
    series_tickers: Optional[List[str]] = None,
    include_active: bool = False,
) -> GroundTruthDataset:
    """
    Fetch ground truth outcomes from all finalized Kalshi contracts.

    Parameters
    ----------
    client : KalshiClientProtocol, optional
        Kalshi API client. Creates one if not provided.
    series_tickers : List[str], optional
        Series tickers to fetch. Uses defaults if not provided.
    include_active : bool
        If True, also fetch active contracts (for price snapshots).

    Returns
    -------
    GroundTruthDataset
        Dataset of settled contract outcomes.
    """
    if client is None:
        client = get_kalshi_client()

    tickers = series_tickers or EARNINGS_SERIES_TICKERS
    all_contracts: List[SettledContract] = []
    all_active_markets: List[Dict] = []

    print(f"Fetching ground truth from {len(tickers)} series tickers...")

    for series_ticker in tickers:
        try:
            # Fetch all markets for this series (all statuses)
            markets = client.get_markets(series_ticker=series_ticker, limit=200)
            if not markets:
                continue

            settled_count = 0
            active_count = 0

            for market in markets:
                status = market.get("status", "").lower()

                if status in ("settled", "finalized", "closed"):
                    contract = _parse_settled_market(market, series_ticker)
                    if contract:
                        all_contracts.append(contract)
                        settled_count += 1
                elif status in ("active", "open") and include_active:
                    all_active_markets.append(market)
                    active_count += 1

            total = settled_count + active_count
            if total > 0:
                print(
                    f"  {series_ticker}: {settled_count} settled, "
                    f"{active_count} active, {len(markets)} total"
                )

        except Exception as e:
            print(f"  Error fetching {series_ticker}: {e}")

    print(f"\nTotal settled contracts: {len(all_contracts)}")

    metadata = {
        "fetched_at": datetime.now().isoformat(),
        "series_tickers_queried": tickers,
        "include_active": include_active,
    }

    return GroundTruthDataset(contracts=all_contracts, metadata=metadata)


def save_ground_truth(dataset: GroundTruthDataset, output_dir: Path) -> Dict[str, Path]:
    """
    Save ground truth dataset to disk.

    Saves:
    - ground_truth.json: Full contract data
    - outcomes.csv: Pivoted outcomes (company x date x word)
    - market_prices.csv: Settlement prices
    - summary.json: Dataset statistics

    Returns dict of file paths.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    paths = {}

    # Full contract data
    contracts_path = output_dir / "ground_truth.json"
    data = {
        "contracts": [c.to_dict() for c in dataset.contracts],
        "metadata": dataset.metadata,
    }
    contracts_path.write_text(json.dumps(data, indent=2, default=str))
    paths["contracts"] = contracts_path
    print(f"Saved {len(dataset.contracts)} contracts to {contracts_path}")

    # Outcomes CSV
    outcomes_df = dataset.outcomes_df
    if not outcomes_df.empty:
        outcomes_path = output_dir / "outcomes.csv"
        outcomes_df.to_csv(outcomes_path)
        paths["outcomes"] = outcomes_path
        print(f"Saved outcomes to {outcomes_path}")

    # Market prices CSV
    prices_df = dataset.market_prices_df
    if not prices_df.empty:
        prices_path = output_dir / "market_prices.csv"
        prices_df.to_csv(prices_path)
        paths["prices"] = prices_path
        print(f"Saved market prices to {prices_path}")

    # Summary
    summary = dataset.summary()
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, default=str))
    paths["summary"] = summary_path
    print(f"Saved summary to {summary_path}")

    return paths


def load_ground_truth(input_dir: Path) -> GroundTruthDataset:
    """Load ground truth dataset from disk."""
    input_dir = Path(input_dir)
    contracts_path = input_dir / "ground_truth.json"

    with open(contracts_path) as f:
        data = json.load(f)

    contracts = [SettledContract(**c) for c in data["contracts"]]
    metadata = data.get("metadata", {})

    return GroundTruthDataset(contracts=contracts, metadata=metadata)


def build_backtest_dataframes(
    dataset: GroundTruthDataset,
    company_ticker: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Build DataFrames suitable for EarningsKalshiBacktester from ground truth.

    Since the Beta-Binomial model only uses historical outcomes (not features),
    the features DataFrame is constructed from the outcomes themselves - each
    word's historical mention count serves as the feature.

    Parameters
    ----------
    dataset : GroundTruthDataset
        Ground truth data.
    company_ticker : str
        Company to build DataFrames for.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
        (features, outcomes, market_prices) DataFrames.
        - features: index=event_date, columns=word names, values=mention counts
        - outcomes: index=event_date, columns=word names, values=0/1
        - market_prices: index=event_date, columns=word names, values=0-1 prices
    """
    ticker_data = dataset.for_ticker(company_ticker)
    if not ticker_data.contracts:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    # Build rows
    rows = []
    for c in ticker_data.contracts:
        if c.event_date:
            rows.append({
                "event_date": c.event_date,
                "word": c.word.lower(),
                "outcome": c.outcome,
                "market_price": c.settlement_price,
            })

    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    # Pivot outcomes
    outcomes = df.pivot_table(
        index="event_date",
        columns="word",
        values="outcome",
        aggfunc="first",
    )
    outcomes.index = pd.to_datetime(outcomes.index)
    outcomes = outcomes.sort_index()
    # Fill NaN with 0 (word not tracked in that period => treat as not mentioned)
    outcomes = outcomes.fillna(0).astype(int)

    # Features: for Beta-Binomial, features = cumulative mention counts
    # This gives the model the historical frequency to estimate probability
    features = outcomes.copy().astype(float)

    # Market prices: settlement prices are NOT pre-trade market prices.
    # Settlement prices (0 or 1) reflect the actual outcome and would leak
    # information into the backtest.  We return an empty DataFrame so the
    # backtester falls back to its 0.5 baseline, which is the honest
    # assumption when we don't have pre-settlement price snapshots.
    market_prices = pd.DataFrame(index=outcomes.index, columns=outcomes.columns, dtype=float)

    return features, outcomes, market_prices
