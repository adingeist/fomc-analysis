"""
Mispricing Detection
====================

This module generates mispricing tables by comparing model probabilities
with Kalshi market prices. It identifies contracts where the model believes
there is significant edge.

Mispricing table includes:
- contract: Contract name
- model_prob: Model probability P(count >= threshold)
- market_prob: Implied probability from Kalshi price
- edge: model_prob - market_prob
- confidence: Measure of model confidence (interval width)
- effective_n: Effective sample size for estimate
- recommendation: "YES" (buy YES if edge > 0), "NO" (buy NO if edge < 0)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import numpy as np


@dataclass
class MispricingResult:
    """Result of mispricing analysis for a single contract."""

    contract: str
    model_prob: float  # P(threshold hit) from model
    market_prob: float  # Implied probability from Kalshi
    edge: float  # model_prob - market_prob
    confidence: float  # Measure of confidence (0-1)
    lower_bound: float  # Lower bound of model interval
    upper_bound: float  # Upper bound of model interval
    recommendation: str  # "YES" or "NO" or "PASS"
    timestamp: Optional[str] = None  # When market price was observed


def compute_mispricing(
    model_predictions: pd.DataFrame,
    market_prices: pd.DataFrame,
    edge_threshold: float = 0.05,
) -> List[MispricingResult]:
    """
    Compute mispricing for all contracts.

    Parameters
    ----------
    model_predictions : pd.DataFrame
        Model predictions with columns: contract, probability, lower_bound,
        upper_bound, uncertainty.
    market_prices : pd.DataFrame
        Market prices with columns: contract, price (0-100 or 0-1 scale).
    edge_threshold : float, default=0.05
        Minimum absolute edge to make a recommendation.

    Returns
    -------
    List[MispricingResult]
        Mispricing results for each contract.
    """
    results = []

    # Merge predictions and prices
    merged = pd.merge(
        model_predictions, market_prices, on="contract", how="inner"
    )

    for _, row in merged.iterrows():
        contract = row["contract"]
        model_prob = row["probability"]
        lower_bound = row.get("lower_bound", model_prob - 0.1)
        upper_bound = row.get("upper_bound", model_prob + 0.1)

        # Convert market price to probability (0-1 scale)
        market_price = row["price"]
        if market_price > 1:  # Assume 0-100 scale
            market_prob = market_price / 100.0
        else:
            market_prob = market_price

        # Compute edge
        edge = model_prob - market_prob

        # Compute confidence (narrower interval = higher confidence)
        interval_width = upper_bound - lower_bound
        confidence = max(0, 1 - interval_width)  # Simple confidence measure

        # Make recommendation
        if abs(edge) < edge_threshold:
            recommendation = "PASS"
        elif edge > 0:
            recommendation = "YES"
        else:
            recommendation = "NO"

        results.append(
            MispricingResult(
                contract=contract,
                model_prob=model_prob,
                market_prob=market_prob,
                edge=edge,
                confidence=confidence,
                lower_bound=lower_bound,
                upper_bound=upper_bound,
                recommendation=recommendation,
            )
        )

    return results


def create_mispricing_table(
    mispricing_results: List[MispricingResult],
    sort_by: str = "edge",
) -> pd.DataFrame:
    """
    Create a mispricing table DataFrame.

    Parameters
    ----------
    mispricing_results : List[MispricingResult]
        Results from compute_mispricing.
    sort_by : str, default="edge"
        Column to sort by. Options: "edge", "abs_edge", "confidence".

    Returns
    -------
    pd.DataFrame
        Mispricing table sorted by specified column.
    """
    rows = []
    for result in mispricing_results:
        rows.append(
            {
                "contract": result.contract,
                "model_prob": result.model_prob,
                "market_prob": result.market_prob,
                "edge": result.edge,
                "abs_edge": abs(result.edge),
                "confidence": result.confidence,
                "lower_bound": result.lower_bound,
                "upper_bound": result.upper_bound,
                "recommendation": result.recommendation,
            }
        )

    df = pd.DataFrame(rows)

    # Sort
    if sort_by == "edge":
        df = df.sort_values("edge", ascending=False)
    elif sort_by == "abs_edge":
        df = df.sort_values("abs_edge", ascending=False)
    elif sort_by == "confidence":
        df = df.sort_values("confidence", ascending=False)

    return df


def get_best_yes_opportunities(
    mispricing_table: pd.DataFrame, top_n: int = 10
) -> pd.DataFrame:
    """
    Get best YES opportunities (positive edge).

    Parameters
    ----------
    mispricing_table : pd.DataFrame
        Mispricing table from create_mispricing_table.
    top_n : int, default=10
        Number of top opportunities to return.

    Returns
    -------
    pd.DataFrame
        Top N contracts with highest positive edge.
    """
    yes_opportunities = mispricing_table[mispricing_table["edge"] > 0]
    return yes_opportunities.sort_values("edge", ascending=False).head(top_n)


def get_best_no_opportunities(
    mispricing_table: pd.DataFrame, top_n: int = 10
) -> pd.DataFrame:
    """
    Get best NO opportunities (negative edge).

    Parameters
    ----------
    mispricing_table : pd.DataFrame
        Mispricing table from create_mispricing_table.
    top_n : int, default=10
        Number of top opportunities to return.

    Returns
    -------
    pd.DataFrame
        Top N contracts with highest negative edge (best NO bets).
    """
    no_opportunities = mispricing_table[mispricing_table["edge"] < 0]
    return no_opportunities.sort_values("edge", ascending=True).head(top_n)


def print_mispricing_report(
    mispricing_table: pd.DataFrame,
    top_n: int = 5,
    timestamp: Optional[str] = None,
) -> None:
    """
    Print a formatted mispricing report.

    Parameters
    ----------
    mispricing_table : pd.DataFrame
        Mispricing table from create_mispricing_table.
    top_n : int, default=5
        Number of top opportunities to show.
    timestamp : Optional[str]
        Timestamp for the report header.
    """
    print("=" * 80)
    print("MISPRICING REPORT")
    if timestamp:
        print(f"Timestamp: {timestamp}")
    print("=" * 80)
    print()

    # Summary statistics
    total_contracts = len(mispricing_table)
    yes_recs = len(mispricing_table[mispricing_table["recommendation"] == "YES"])
    no_recs = len(mispricing_table[mispricing_table["recommendation"] == "NO"])
    pass_recs = len(mispricing_table[mispricing_table["recommendation"] == "PASS"])

    print(f"Total contracts analyzed: {total_contracts}")
    print(f"YES recommendations: {yes_recs}")
    print(f"NO recommendations: {no_recs}")
    print(f"PASS (no edge): {pass_recs}")
    print()

    # Best YES opportunities
    print("-" * 80)
    print(f"TOP {top_n} YES OPPORTUNITIES (Buy YES - Model believes higher prob)")
    print("-" * 80)
    yes_opps = get_best_yes_opportunities(mispricing_table, top_n)
    if len(yes_opps) > 0:
        print(yes_opps[["contract", "model_prob", "market_prob", "edge", "confidence"]].to_string(index=False))
    else:
        print("No YES opportunities found.")
    print()

    # Best NO opportunities
    print("-" * 80)
    print(f"TOP {top_n} NO OPPORTUNITIES (Buy NO - Model believes lower prob)")
    print("-" * 80)
    no_opps = get_best_no_opportunities(mispricing_table, top_n)
    if len(no_opps) > 0:
        # Show with absolute edge for easier reading
        no_display = no_opps.copy()
        no_display["edge_magnitude"] = -no_display["edge"]
        print(no_display[["contract", "model_prob", "market_prob", "edge_magnitude", "confidence"]].to_string(index=False))
    else:
        print("No NO opportunities found.")
    print()

    print("=" * 80)


def load_market_prices_from_csv(csv_path: Path) -> pd.DataFrame:
    """
    Load market prices from CSV file.

    Expected CSV format:
    contract,price,timestamp
    "Inflation 40+",65.5,2025-01-15T10:00:00
    "Price 15+",42.0,2025-01-15T10:00:00

    Parameters
    ----------
    csv_path : Path
        Path to CSV file with market prices.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: contract, price, timestamp.
    """
    df = pd.read_csv(csv_path)
    required_cols = {"contract", "price"}

    if not required_cols.issubset(df.columns):
        raise ValueError(f"CSV must contain columns: {required_cols}")

    return df


def load_market_prices_from_kalshi_api(
    contracts: List[str],
    api_key: Optional[str] = None,
) -> pd.DataFrame:
    """
    Load current market prices from Kalshi API.

    Parameters
    ----------
    contracts : List[str]
        List of contract names to fetch prices for.
    api_key : Optional[str]
        Kalshi API key. If None, will try to load from environment.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: contract, price, timestamp.
    """
    # TODO: Implement Kalshi API integration
    # This would use the kalshi_api module to fetch current prices
    raise NotImplementedError("Kalshi API integration not yet implemented")
